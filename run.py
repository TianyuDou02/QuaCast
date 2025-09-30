import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import os.path as osp
import math
import time
import argparse
import logging 
import yaml
import cProfile
from tqdm import tqdm
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from datasets.get_datasets import get_dataset
from utils.metrics import Evaluator
from utils.tools import print_log, cycle, show_img_info

# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "YOUR_WANDB"
# os.environ["WANDB_SILENT"] = "true"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"

def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone', default='past',  type=str,                 help='backbone model for deterministic prediction')
    
    parser.add_argument("--seed",           type=int,   default=42,              help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='basic_exps',   help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default=None,           help="additional note for experiment")


    # --------------- Dataset ---------------
    parser.add_argument("--dataset",        type=str,   default='shanghai',        help="dataset name")
    parser.add_argument("--img_size",       type=int,   default=128,            help="image size")
    parser.add_argument("--img_channel",    type=int,   default=1,              help="channel of image")
    parser.add_argument("--seq_len",        type=int,   default=25,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",      type=int,   default=5,              help="number of frames to input")
    parser.add_argument("--frames_out",     type=int,   default=20,             help="number of frames to output")    
    parser.add_argument("--num_workers",    type=int,   default=4,              help="number of workers for data loader")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=5e-4,            help="learning rate")
    parser.add_argument("--lr-beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr-beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=0.0,             help="l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='cosine',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps",   type=int,   default=1000,            help="warmup steps")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=1,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=3,              help="batch size")
    parser.add_argument("--epochs",         type=int,   default=100,              help="number of epochs")
    parser.add_argument("--training_steps", type=int,   default=1,          help="number of training steps")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default=None,            help="resumed checkpoint milestone")
    
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",                 help="evaluation mode")
    parser.add_argument("--wandb_state",    type=str,   default='disabled',      help="wandb state config")
    
    parser.add_argument("--speed_test",     type=int,   default=0,               help="speed  mode")
    args = parser.parse_args()
    return args


class Runner(object):
    
    def __init__(self, args):
        self.args = args
        self._preparation()
        # Config DDP kwargs from accelerate
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=self.log_path
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        
        self.accelerator = Accelerator(
            project_config  =   project_config,
            kwargs_handlers =   [ddp_kwargs, process_kwargs],
            mixed_precision =   self.args.mixed_precision,
            log_with        =   'wandb'
        )
        
        # Config log tracker 'wandb' from accelerate
        self.accelerator.init_trackers(
            project_name=self.exp_name,
            config=self.args.__dict__,
            init_kwargs={"wandb": 
                {
                "mode": self.args.wandb_state,
                # 'resume': self.args.ckpt_milestone
                }
                         }   # disabled, online, offline
        )
        
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
    
        print_log(self.accelerator.state, self.is_main)
        
        self._load_data()
        self._build_model()
        self._build_optimizer()
        
        # distributed ema for parallel sampling

        self.model, self.optimizer,  self.scheduler, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, self.scheduler,
            self.train_loader, self.valid_loader, self.test_loader
        )
        
        self.train_dl_cycle = cycle(self.train_loader)
        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)
            # print_log(show_img_info(sample), self.is_main)
            
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = 'Single' + self.args.backbone
        self.exp_name   = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"
        
        cur_dir         = os.path.dirname(os.path.abspath(__file__))
        
        self.exp_dir    = osp.join(cur_dir, 'Exps', self.args.exp_dir, self.exp_name)        
        self.ckpt_path  = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path  = osp.join(self.exp_dir, 'test_samples')
        self.log_path   = osp.join(self.exp_dir, 'logs')
        self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        exp_params      = self.args.__dict__
        params_path     = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # filemode='a',
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'log.log')),
                # logging.StreamHandler()
            ]
        )
        
    def _load_data(self):
        # =================================
        # Get Train/Valid/Test dataloader among datasets 
        # =================================

        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            # data_path=self.args.data_path,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
        )
        
        self.visiual_save_fn = color_save_fn
        self.thresholds      = THRESHOLDS
        self.scale_value     = PIXEL_SCALE
        
        if self.args.dataset != 'sevir':
            # preload big batch data for gradient accumulation
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size*self.args.grad_acc_step, shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
            
        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",
                  self.is_main)
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}",
                  self.is_main)
        
    def _build_model(self):
        # =================================
        # import and create different models given model config
        # =================================
        if self.args.backbone == 'quacast':
            from models.QuaCast import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
            }
            model = get_model(**kwargs)
        
        
        else:
            raise ValueError(f"Invalid model name: {self.args.backbone}")
        
        self.model = model
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)        
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)


    def _build_optimizer(self):
        # =================================
        # Calcutate training nums and config optimizer and learning schedule
        # =================================
        num_steps_per_epoch = len(self.train_loader)
        num_epoch = math.ceil(self.args.training_steps / num_steps_per_epoch)
        
        self.global_epochs = max(num_epoch, self.args.epochs)
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch
        
        self.cur_step, self.cur_epoch = 0, 0

        warmup_steps = self.args.warmup_steps

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )
        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=self.global_steps,
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps , 
                num_training_steps=self.global_steps,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    self.args.scheduler
            )
        )
            
        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"optimizer: {self.optimizer} with init lr: {self.args.lr}")
        
    
    def save(self,path=None):
        # =================================
        # Save checkpoint state for model and ema
        # =================================
        if not self.is_main:
            return
        
        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),

        }
        if path is None:
            path = osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt")
        torch.save(data, path)
        # print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        
        
    def load(self, milestone):
        # =================================
        # load model checkpoint
        # =================================        
        device = self.accelerator.device
        
        if '.pt' in [milestone]:
            data = torch.load(milestone, map_location=device)
        else:
            
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.model = self.accelerator.prepare(model)
        
        self.optimizer.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])
        
        if self.is_main:
            self.ema.load_state_dict(data['ema'])

        # self.cur_epoch = data['epoch']
        # self.cur_step = data['step']
        print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
    
    def train(self):
        # set global step as traing process
        pbar = tqdm(
            initial=self.cur_step,
            total=self.global_steps // torch.cuda.device_count(),
            disable=not self.is_main,
        )
        start_epoch = self.cur_epoch
        min_loss = float('inf')
        best_model_path = None

        for epoch in range(start_epoch, self.global_epochs):
            self.cur_epoch = epoch
            self.model.train()
            epoch_loss = 0
            
            for i, batch in enumerate(self.train_loader):
                # train the model with mixed_precision
                with self.accelerator.autocast(self.model):

                    loss_dict = self._train_batch(batch)
                    total_loss = loss_dict["total_loss"]
                    epoch_loss += float(total_loss.detach())
                    self.accelerator.backward(loss_dict['total_loss'])
                    
                    if self.cur_step == 0:
                        # training process check
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)   
    
                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()
                
                # record train info
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = dict()
                log_dict['lr'] = lr
                for k,v in loss_dict.items():
                    log_dict[k] = v.item()
                self.accelerator.log(log_dict, step=self.cur_step)
                pbar.set_postfix(**log_dict)   
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch // torch.cuda.device_count()}"
                pbar.set_description(state_str)
                
                # update ema param and log file every 20 steps
                if i % 20 == 0:
                    logging.info(state_str+'::'+str(log_dict))
                self.ema.update()

                self.cur_step += 1
                pbar.update(1)
                
                # do santy check at begining
                if self.cur_step == 1:
                    """ santy check """
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon= self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            if self.is_main:
                                for i in range(radar_ori.shape[0]):
                                    self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(self.sanity_path, f"{i}/vil"),data_type='vil')

                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            if avg_epoch_loss < min_loss:
                min_loss = avg_epoch_loss
                if best_model_path and osp.exists(best_model_path):
                    os.remove(best_model_path)
                ckpt_name = f'step={self.cur_step}_loss={avg_epoch_loss}'
                # best_model_path = osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt")
                best_model_path = osp.join(self.ckpt_path, f"ckpt-{ckpt_name}.pt")
                self.save(best_model_path)
                self.save(osp.join(self.ckpt_path,"ckpt-best.pt"))
                print_log(f"Saved new best model with loss {min_loss} at epoch {epoch}", self.is_main)
            else: print_log(f"{avg_epoch_loss} is not the best one", self.is_main)
            print_log(f" ========= Finisth one Epoch ==========", self.is_main)
        self.save(osp.join(self.ckpt_path,"ckpt-last.pt"))
        print_log(f"Saved the last model with loss {min_loss}", self.is_main)
        self.accelerator.end_training()
        # with torch.no_grad():
        #     self.test_samples(ckpt_name, do_test=True)
        
        return ckpt_name
        
    def _get_seq_data(self, batch):
        # frame_seq = batch['vil'].unsqueeze(2).to(self.device)
        return batch      # [B, T, C, H, W]
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        # multi GPUs
        _, loss = self.model.module.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
        # single GPU
        #  _, loss = self.model.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
      
        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        return {'total_loss': loss}
        
    
    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False):
        # multi GPUs
        sample_fn = self.ema.ema_model.predict if use_ema else self.model.module.predict
        # single GPU
        # sample_fn = self.ema.ema_model.predict if use_ema else self.model.predict

        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        radar_pred, _ = sample_fn(radar_input,compute_loss=False)
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()

        return radar_gt, radar_pred
    
    
    def test_samples(self, milestone, do_test=False):
        # init test data loader
        data_loader = self.test_loader if do_test else self.valid_loader
        # init sampling method
        self.model.eval()
        # init test dir config
        cnt = 0
        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)
        # if self.is_main:
        eval = Evaluator(
            seq_len=self.args.frames_out,
            value_scale=self.scale_value,
            thresholds=self.thresholds,
            save_path=save_dir,
        )
        # print_log(f"Test data loader length: {len(data_loader)}", self.is_main)
        # for batch in data_loader:
        #     print_log(f"Batch shape: {batch.shape}", self.is_main)
        #     break
        # start test loop
        for batch in tqdm(data_loader,desc='Test Samples', disable=not self.is_main):
            # sample
            
            radar_ori, radar_recon= self._sample_batch(batch)
            # evaluate result and save
            eval.evaluate(radar_ori, radar_recon)
            if self.is_main:
                for i in range(radar_ori.shape[0]):
                    self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(save_dir, f"{cnt}-{i}/vil"),data_type='vil')

            self.accelerator.wait_for_everyone()
            # cnt += 1
            # if cnt > 10:
            #     break
        # test done
        if self.is_main:
            res = eval.done()
            print_log(f"Test Results: {res}")
            print_log("="*30)
    @torch.inference_mode()
    def benchmark_inference(self, num_warmup=20, num_steps=100, use_ema=False, batch_size_override=None):
        """
        评估推理速度（多卡下统计全局吞吐量与延迟）
        - num_warmup: 预热步数（不计时）
        - num_steps: 计时步数
        - use_ema: 是否使用 EMA 模型进行预测
        - batch_size_override: 指定一个固定 batch size 做合成基准（不经过 DataLoader）
        说明：
        - 吞吐量 (sequences/sec)：每次调用 predict 的序列数（batch_size）/ 时间
        - 帧吞吐量 (frames/sec)：sequences/sec × T_out
        - 平均序列延迟 (ms/seq)：1000 / sequences/sec（单卡）；全局则用总时间 / 全局样本数
        """
        self.model.eval()

        # 选择推理函数（多卡需要 .module）
        sample_fn = self.ema.ema_model.predict if use_ema else self.model.module.predict

        # 固定输入形状下可加速
        torch.backends.cudnn.benchmark = True

        # 准备一个批次（从 valid_loader 取一个，或用合成数据）
        if batch_size_override is None:
            try:
                batch = next(iter(self.valid_loader))
            except StopIteration:
                raise RuntimeError("valid_loader 为空，无法做基准测试。")
            radar_batch = self._get_seq_data(batch).to(self.device, non_blocking=True)
            B = radar_batch.shape[0]
            frame_in = self.args.frames_in
            frames_in = radar_batch[:, :frame_in]
        else:
            # 合成数据：避免 DataLoader I/O 干扰
            B = batch_size_override
            C, H, W = self.args.img_channel, self.args.img_size, self.args.img_size
            T_in = self.args.frames_in
            frames_in = torch.randn(B, T_in, C, H, W, device=self.device)

        # 预热（不计时）
        for _ in range(num_warmup):
            _ = sample_fn(frames_in, compute_loss=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # 正式计时
        start = time.perf_counter()
        for _ in range(num_steps):
            _ = sample_fn(frames_in, compute_loss=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start  # 当前 rank 的耗时

        # ====== 多卡汇总（用最大耗时作为全局耗时，样本数按世界大小相乘）======
        world_size = self.accelerator.num_processes
        device = self.device
        import torch.distributed as dist

        t = torch.tensor([elapsed], device=device)
        if world_size > 1 and dist.is_initialized():
            # 全局最大耗时（所有 rank 同步）
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        global_elapsed = t.item()

        # 当前 rank 每秒序列吞吐
        local_seqs_per_s = (num_steps * B) / elapsed
        # 全局吞吐：总处理样本（num_steps * B * world_size）/ 全局最大耗时
        global_seqs_per_s = (num_steps * B * world_size) / global_elapsed

        # 换算帧吞吐（按 T_out）
        frames_out = self.args.frames_out
        local_frames_per_s = local_seqs_per_s * frames_out
        global_frames_per_s = global_seqs_per_s * frames_out

        # 平均延迟（以全局为准）
        global_ms_per_seq = (global_elapsed / (num_steps * B * world_size)) * 1000.0

        if self.is_main:
            print_log("========== Inference Benchmark ==========", self.is_main)
            print_log(f"Model: {self.model_name} | EMA: {use_ema}", self.is_main)
            print_log(f"World Size: {world_size} | Batch Size/Rank: {B}", self.is_main)
            print_log(f"Timed Steps: {num_steps} | Warmup Steps: {num_warmup}", self.is_main)
            print_log(f"Elapsed (rank max): {global_elapsed:.4f}s", self.is_main)
            print_log(f"Throughput (seq/s): local {local_seqs_per_s:.2f} | global {global_seqs_per_s:.2f}", self.is_main)
            print_log(f"Throughput (frames/s): local {local_frames_per_s:.2f} | global {global_frames_per_s:.2f}", self.is_main)
            print_log(f"Latency (ms/seq, global): {global_ms_per_seq:.3f}", self.is_main)

        return {
            "elapsed_global_s": global_elapsed,
            "seqs_per_s_local": local_seqs_per_s,
            "seqs_per_s_global": global_seqs_per_s,
            "frames_per_s_local": local_frames_per_s,
            "frames_per_s_global": global_frames_per_s,
            "latency_ms_per_seq_global": global_ms_per_seq,
            "batch_size_per_rank": B,
            "world_size": world_size,
        }
        
    def check_milestones(self, target_ckpt=None):

        # mils_paths = os.listdir(self.ckpt_path)
        # milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        # print_log(f"milestones: {milestones}", self.accelerator.is_main_process)
        
        if target_ckpt is not None:
            
            self.load(target_ckpt)
            print(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples(saved_dir_name, do_test=True)
            return
        
        # for m in range(0, len(milestones), 1):
        #     self.load(milestones[m])
        #     self.test_samples(milestones[m], do_test=True)
            
def main():
    args = create_parser()
    exp = Runner(args)
    if not args.eval:
        ckpt_milestone = "best"
        exp.train()
    else:
        if args.ckpt_milestone:
            ckpt_milestone = args.ckpt_milestone
        else:
            ckpt_milestone = "best"
    # exp.check_milestones(target_ckpt=args.ckpt_milestone)
    if args.speed_test == 1:
        exp.benchmark_inference(num_warmup=50, num_steps=200, use_ema=True, batch_size_override=exp.args.batch_size)
    else:
        exp.check_milestones(target_ckpt="best")

if __name__ == '__main__':
    # 测试代码各模块执行效率
    # pip install graphviz
    # pip install gprof2dot
    # gprof2dot -f pstats train.profile | dot -Tpng -o result.png
    # cProfile.run('main()', filename='train.profile', sort='cumulative')
    main()
