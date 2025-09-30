import sys
sys.path.append('/media/root1/B3D12CBDE981113E1/dty/DiffCast-main/models/FourierQuaternion')
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from module import fft2d, ResnetBlock, wavelet_quaternion, fourier_quaternion, pure_quaternion, Decoder, MidIncepNet, LSEncoder
from quaternion_layers import QuaternionConv, QuaternionLinear, QuaternionTransposeConv


def phase_loss(phi1, phi2, weight_type='freq', sigma=10.0, amp1=None, amp2=None):
    # Ensure inputs are float tensors
    phi1 = phi1.float()
    phi2 = phi2.float()
    # Compute phase difference loss: 1 - cos(phi1 - phi2)
    phase_diff = 1 - torch.cos(phi1 - phi2 + 1e-8)
    # Compute weights
    if weight_type == 'freq':
        # Frequency-based weights (Gaussian decay from center)
        B, T, C, H, W = phi1.shape
        u = torch.linspace(-0.5, 0.5, H, device=phi1.device)
        v = torch.linspace(-0.5, 0.5, W, device=phi1.device)
        U, V = torch.meshgrid(u, v, indexing='ij')
        weights = torch.exp(-(U**2 + V**2) / (2 * sigma**2))
        weights = weights.view(1, 1, 1, H, W).expand(B, T, C, H, W)
        weights = weights / (weights.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    elif weight_type == 'amp':
        # Amplitude-based weights
        if amp1 is None or amp2 is None:
            raise ValueError("amp1 and amp2 must be provided for amplitude weighting")
        weights = torch.minimum(amp1, amp2)
        weights = weights / (weights.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    else:
        # Uniform weights
        weights = torch.ones_like(phase_diff) / (phase_diff.numel() + 1e-8)
    
    # Compute weighted loss
    loss = torch.sum(weights * phase_diff)
    
    return loss

def amplitude_loss(amp1, amp2, weight):
     return weight* nn.MSELoss()(amp1, amp2)

def fourier_loss(frames_pred, framed_gt, amplitude_weight=0.001, phase_weight=0.001, weight_type='amp', sigma=10.0):
    amp_pred, pha_pred = fft2d(frames_pred)
    amp_gt, pha_gt = fft2d(framed_gt)
    pha_loss_value = phase_loss(pha_pred, pha_gt, weight_type=weight_type, sigma=sigma, amp1=amp_pred, amp2=amp_gt)
    amp_loss_value = amplitude_loss(amp_pred, amp_gt, weight=amplitude_weight)
    loss = phase_weight*pha_loss_value + amp_loss_value
    return loss

def loss_gradient_difference(real_image, generated): 
        m,n = real_image.shape[-2:]
        true_x_shifted_right = real_image[:,:,:,1:,:]
        true_x_shifted_left = real_image[:,:,:,:-1,:]
        true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

        generated_x_shift_right = generated[:,:,:,1:,:]
        generated_x_shift_left = generated[:,:,:,:-1,:]
        generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

        difference_x = true_x_gradient - generated_x_griednt

        # loss_x_gradient = (torch.sum(difference_x)**2)/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)
        loss_x_gradient = torch.nn.MSELoss()(true_x_gradient, generated_x_griednt)
        

        true_y_shifted_right = real_image[:,:,:,:,1:]
        true_y_shifted_left = real_image[:,:,:,:,:-1]
        true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

        generated_y_shift_right = generated[:,:,:,:,1:]
        generated_y_shift_left = generated[:,:,:,:,:-1]
        generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

        difference_y = true_y_gradient - generated_y_griednt
        loss_y_gradient = torch.nn.MSELoss()(true_y_gradient, generated_y_griednt)
        
        # igdl = (loss_x_gradient + loss_y_gradient)/(m+n-1)
        igdl = (loss_x_gradient + loss_y_gradient)
        return igdl

def ExGradLoss(pred, target, up_th = 0.95, down_th = 0.05, lamda = 1.0):
        # Loss of the gradient of the extreme value
        B, T, C, H, W = pred.shape
        # Get the 90% and 10% quantiles in target as the thresholds for extreme maximum and minimum values, denoted as tar_up and tar_down
        tar_up = torch.quantile(target.view(B, T, C, H*W), q=up_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,T,C,1,1

        tar_down = torch.quantile(target.view(B, T, C, H*W), q=down_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,T,C,1,1

        target_up_area = F.relu(target-tar_up) # The part of target that is greater than tar_up
        target_down_area = -F.relu(tar_down-target) # The part of target that is smaller than tar_down
        pred_up_area = F.relu(pred-tar_up) # The part of pred that is greater than tar_up
        pred_down_area = -F.relu(tar_down-pred) # The part of pred that is smaller than tar_down

        loss_up = loss_gradient_difference(target_up_area, pred_up_area)
        loss_down = loss_gradient_difference(target_down_area, pred_down_area)
        ex_loss = (loss_up + loss_down)/(1-up_th+down_th)
        return lamda*ex_loss

class STNorm(nn.Module):
    def __init__(self, T_in, T_out, eps=1e-5,):
        super(STNorm, self).__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.eps = eps
    def forward(self, x, mode='norm', mean=None, std=None):
        assert mode in ['norm', 'denorm'], "Mode must be 'norm' or 'denorm'"
        if mode == 'norm':
            mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
            std = x.std(dim=(1, 2, 3, 4), keepdim=True)
            x_norm = (x - mean) / (std + self.eps)
            return x_norm, mean, std
        elif mode == 'denorm':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for denormalization")
            x_denorm = x * (std + self.eps) + mean
            return x_denorm


class FourierQuaternionModel(nn.Module):
    def __init__(self, in_shape, T_in, T_out,
                amp_weight = 0.01, pha_weight = 0.01,
                aweight_stop_steps=10000,
                 **kwargs):
        super(FourierQuaternionModel, self).__init__()
        C, H, W = in_shape
        self.T_in = T_in
        self.T_out = T_out
        self.amp_weight = amp_weight
        self.pha_weight = pha_weight
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate = amp_weight / aweight_stop_steps

        depth_S = 4
        depth_T = 8
        self.iters = 0
        self.encoder = LSEncoder(C*4, 64, depth_S, 3, 4)
        self.decoder = Decoder(64, C*4, depth_S, 3, rotation=False)
        self.mid_incep = MidIncepNet(T_in*64, 256, depth_T, channel_out= T_out*64, rotation=False, groups=8)
        self.revlnorm = STNorm(T_in=T_in, T_out=T_out)
        self.mix = nn.Sequential(
             ResnetBlock(C*4, 64),
             ResnetBlock(64, 64),
             nn.Conv2d(64, C, 1)
        )

        self.proj_skip = nn.Conv2d(64*T_in, 64*T_out, 1)

    def forward(self, x,):
        B, T, C, H, W = x.shape
        x, in_mean, in_std = self.revlnorm(x)

        quaternion_x = pure_quaternion(x)
        embed, skip = self.encoder(quaternion_x)
        _, C_, _, _ = skip.shape
        skip = rearrange(skip, '(b t) c h w -> b (t c) h w', b=B, t=T)
        skip = self.proj_skip(skip)
        skip = rearrange(skip, 'b (t c) h w -> (b t) c h w', b=B, c=C_)
        _, _, C_, H_, W_ = embed.shape
        hid = self.mid_incep(embed)
        hid = rearrange(hid, 'b t c h w -> (b t) c h w')
        de = self.decoder(hid, skip)

        de = self.mix(de)
        output = rearrange(de, '(b t) c h w -> b t c h w', b=B)
        output = self.revlnorm(output, mode='denorm', mean=in_mean, std=in_std)
        return output

    def predict(self, frames_in, frames_gt=None, compute_loss=False, **kwargs):
        self.iters += 1
        frames_pred = self.forward(frames_in)
        if compute_loss:
            if self.iters < self.aweight_stop_steps:
                self.amp_weight -= self.sampling_changing_rate
                self.pha_weight -= self.sampling_changing_rate
            else:
                self.amp_weight  = 0.
                self.pha_weight  = 0.
            loss = nn.MSELoss()(frames_pred, frames_gt) \
                + 0.1*ExGradLoss(frames_pred, frames_gt) \
                + fourier_loss(frames_pred, frames_gt, amplitude_weight=self.amp_weight, phase_weight=self.pha_weight, weight_type='amp')\
                
                
        else:
            loss = None
        return frames_pred, loss

def get_model(in_shape, T_in, T_out, **kwargs):
    return FourierQuaternionModel(in_shape, T_in=T_in, T_out=T_out, **kwargs)

if __name__ == "__main__":
    
    in_shape = (1, 128, 128)
    model = get_model(in_shape, T_in=5, T_out=20).to('cuda')
    frames_in = torch.randn(3, 5, 1, 128, 128).to('cuda')
    frames_gt = torch.randn(3, 20, 1, 128, 128).to('cuda')
    
    frames_pred, loss = model.predict(frames_in=frames_in, frames_gt=frames_gt, compute_loss=True)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None :
            print(name)
    print(frames_pred.shape)
    print(loss)
    total_params = sum(p.numel() for p in model.parameters())  
    total_size = total_params  / (1000 ** 2)  
    print(f"Model size: {total_size:.4f}M")

