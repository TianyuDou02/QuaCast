import sys
sys.path.append('/media/root1/B3D12CBDE981113E/dty/DiffCast-main/models/FourierQuaternion')
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from quaternion_layers import QuaternionConv, QuaternionTransposeConv
from pytorch_wavelets import DWTForward
import numpy as np
import pywt
from scipy.stats import levy_stable

def fft2d(x):
    fft_out = torch.fft.fft2(x, dim=(-2, -1))  # 沿空间维度(H, W)做FFT
    amplitude = torch.abs(fft_out)
    phase = torch.angle(fft_out)
    return amplitude, phase

def ifft2d(amplitude, phase):
    complex_out = amplitude * torch.exp(1j * phase)
    return torch.fft.ifft2(complex_out, dim=(-2, -1)).real

def pure_quaternion(x):
    return torch.cat([x, x, x, x], dim=2)  # (B, T, C*4, H, W)

def fourier_quaternion(x):
    B, T, C, H, W = x.shape
    amp, phase = fft2d(x)
    mean = torch.mean(x, dim=(2, 3, 4), keepdim=True).repeat(1, 1, C, H, W)
    # quaternion representation
    zero = torch.zeros_like(x)
    x_quat = torch.cat([zero, amp, phase, x], dim=2)  # (B, T, C*4, H, W)
    return x_quat

def wavelet_quaternion(x, wavelet='haar', J=1):
    device = x.device
    B, T, C, H, W = x.shape
    x = rearrange(x, 'b t c h w -> (b t) c h w')
    x_np = x.detach().cpu().numpy()
    coeffs_list = []
    for i in range(B*T):  # Batch dimension
        coeffs = pywt.swt2(x_np[i, 0], wavelet=wavelet, level=J)
        if isinstance(coeffs, list):
            cA, H = coeffs[0] 
            cH, cV, cD = H
        else:  # Handle tuple output
            cA, (cH, cV, cD) = coeffs
        coeffs = np.stack([cA, cH, cV, cD], axis=0) 
        coeffs_list.append(coeffs)
    coeffs_np = np.stack(coeffs_list) 
    coeffs_tensor = torch.tensor(coeffs_np, requires_grad=False, device=device)
    coeffs_tensor = rearrange(coeffs_tensor, '(b t) c h w -> b t c h w', b=B, t=T)
    return coeffs_tensor

def iWavelet(x, wavelet='haar', J=1):
    device = x.device
    B, T, C, H, W = x.shape
    x = rearrange(x, 'b t c h w -> (b t) c h w')
    x_np = x.detach().cpu().numpy()
    coeffs_list = []
    for i in range(B*T):  # Batch dimension
        coeffs = pywt.iswt2([x_np[i, 0:1],(x_np[i, 1:2], x_np[i, 2:3], x_np[i, 3:])], wavelet=wavelet)
        coeffs_list.append(coeffs)
    coeffs_np = np.stack(coeffs_list) 
    coeffs_tensor = torch.tensor(coeffs_np, requires_grad=False, device=device)
    coeffs_tensor = rearrange(coeffs_tensor, '(b t) c h w -> b t c h w', b=B, t=T)
    return coeffs_tensor



class DoubleConv(nn.Module):

    def   __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            QuaternionConv(in_channels, mid_channels, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            QuaternionConv(mid_channels, out_channels, kernel_size=kernel, padding=kernel//2),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            QuaternionConv(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels // 2)
        else:
            self.up = QuaternionTransposeConv(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_S(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, kernel=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, mid_channels=in_channels)
        else:
            self.up = QuaternionTransposeConv(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = QuaternionConv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ConvUnet(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
        super(ConvUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c * 1, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear)
        self.outc = OutConv(base_c * 1, n_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=T, c=C)
        return x

class fr_te_fusion(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(fr_te_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', c=C//2)
        return x
        
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', groupnorm=True):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding = kernel_size//2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros'): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = rearrange(x, 'b (g c) h w -> b g c h w', g=groups)
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C // groups, groups, H, W)
    x = rearrange(x, 'b c g h w -> b (c g) h w')  # (B, C, H, W)
    return x

class ChannelShuffleNet(nn.Module):
    def __init__(self, C, groups=8, kernel_size=3, rotation=False):
        super(ChannelShuffleNet, self).__init__()
        self.conv1 = QuaternionConv(C, C, 1, padding=0, rotation=rotation, groups=groups)
        self.conv = QuaternionConv(C, C, kernel_size=kernel_size, padding=kernel_size//2, rotation=rotation, groups=groups)
        self.norm = nn.GroupNorm(groups, C)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = channel_shuffle(x, groups=self.norm.num_groups)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x+res

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True,
                 rotation=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                QuaternionConv(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, rotation=rotation, dilatation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
             self.conv = QuaternionConv(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, rotation=rotation, )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            # trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True,
                 rotation=False):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace, rotation=rotation)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True,
                 operation='convolution2d',
                 rotation=False):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = QuaternionConv(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, rotation=rotation, operation=operation)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8, operation='convolution2d', rotation=False):        
        super(gInception_ST, self).__init__()
        self.conv1 = QuaternionConv(C_in, C_hid, kernel_size=1, stride=1, padding=0, rotation=rotation, operation=operation)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True, operation=operation, rotation=rotation))
        
        self.layers = nn.Sequential(*layers)
        self.shuffle = ChannelShuffleNet(C_out, groups=groups, kernel_size=3, rotation=rotation)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        y = self.shuffle(y)
        return y



def sampling_generator(N, reverse=False):
    samplings = [False,True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]
    
class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True, rotation=False):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Sequential(
                ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace, rotation=rotation),
                ResnetBlock(C_hid, C_hid, groups=8, kernel_size=3,)
            ),
            *[nn.Sequential(ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace, rotation=rotation),
                     ResnetBlock(C_hid, C_hid, groups=8, kernel_size=3))
                 for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True, rotation=False):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[nn.Sequential(
                    ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace, rotation=False),
                     ResnetBlock(C_hid, C_hid, groups=8)
                     ) for s in samplings[:-1]],
             nn.Sequential(ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace, rotation=False),
                     ResnetBlock(C_hid, C_hid, groups=8))
        )
        self.readout = QuaternionConv(C_hid, C_out, 1, rotation=rotation)

    def forward(self, hid, enc1=None):
        
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y

class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, channel_out=None, incep_ker=[3,5,7,11], groups=8, rotation=False, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        if channel_out is None:
            channel_out = channel_in
        self.N2 = N2
        self.groups = groups
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups, rotation=rotation)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups, rotation=rotation))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups, rotation=rotation))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups, rotation=rotation)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups, rotation=rotation))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_out,
                              incep_ker=incep_ker, groups=groups, rotation=rotation))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # x = x.reshape(B, T*C, H, W)
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        # y = z.reshape(B, T, C, H, W)
        y = rearrange(z, 'b (t c) h w -> b t c h w', c=C )
        return y


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', groupnorm=True):
        super(Block, self).__init__()
        self.proj = QuaternionConv(dim, dim_out, kernel_size=kernel_size, padding = kernel_size//2,)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros'): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, kernel_size=kernel_size,)
        self.block2 = Block(dim_out, dim_out, groups = groups, kernel_size=kernel_size,)
        self.res_conv = QuaternionConv(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)



class LSBlock(nn.Module):
    def __init__(
            self, C_in, C_out,
            groups=8, kernel_size=3, down=False, up = False
    ):
        self.down = down
        self.up = up
        super(LSBlock, self).__init__()
        self.block1 = ResnetBlock(C_in, C_out, groups=1, kernel_size=kernel_size)
        if down:
            self.sampling1 = QuaternionConv(C_in, C_out, kernel_size=2, stride=2, padding=0)
        elif up:
            self.sampling1 = QuaternionTransposeConv(C_in, C_out, kernel_size=2, stride=2, padding=0)
        else:
            self.sampling1 = nn.Identity()
        
        self.blocks2 = nn.Sequential(
            ResnetBlock(C_out, C_out, groups=groups, kernel_size=kernel_size),
            ResnetBlock(C_out, C_out, groups=groups, kernel_size=kernel_size)
        )
        if down:
            self.sampling2 = QuaternionConv(C_in, C_out, kernel_size=2, stride=2, padding=0)
        elif up:
            self.sampling2 = QuaternionTransposeConv(C_in, C_out, kernel_size=2, stride=2, padding=0)
        else:
            self.sampling2 = nn.Identity()
        self.fusion = ResnetBlock(C_out*2, C_out, groups=groups, kernel_size=kernel_size)
    def forward(self, x):
        # B, T, C, H, W = x.shape
        x = self.block1(x)
        sampling1 = self.sampling1(x)
        sampling2 = self.sampling2(self.blocks2(x))
        x = torch.cat([sampling1, sampling2], dim=1)
        x = self.fusion(x)
        return x

class LSEncoder(nn.Module):
    def __init__(self, C_in, C_out, N_S,
                 kernel_size=3, groups=8,):
        super(LSEncoder, self).__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            LSBlock(C_in, C_out, groups=groups, kernel_size=kernel_size, down=samplings[0]),
            *[LSBlock(C_out, C_out, groups=groups, kernel_size=kernel_size, down=samplings[i])
              for i in range(1, N_S)]
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        skips = []
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.enc[0](x)
        # skips.append(x)
        skip = x
        for i in range(1, len(self.enc)):
            x = self.enc[i](x)
        # x = self.enc(skip)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        # skip = rearrange(skip, '(b t) c h w -> b t c h w', b=B, t=T)
        return x, skip

