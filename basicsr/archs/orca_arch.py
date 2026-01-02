# from torch.autograd import Variable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs
from collections import OrderedDict

from basicsr.archs.arch_util import LayerNorm, _make_pretrained_efficientnet_lite3, _make_scratch, flow_warp
from basicsr.archs.Depth_Anything.depth_anything.dpt import DepthAnything
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class FlowPromptGenerationModule(nn.Module):
    """
        Generating an input-conditioned prompt.
    """
    def __init__(self, embed_dim=64, prompt_dim=96, prompt_len=5, prompt_size=96, align=False):
        super(FlowPromptGenerationModule, self).__init__()
        self.align = align
        self.prompt_dim = prompt_dim

        self.linear_proj = nn.Linear(embed_dim, prompt_len) # project feature embeddings to prompt dims

    def forward(self, x):
        b, c, h, w = x.shape                    # torch.Size([2, 128, 64, 64])
        emb = x.mean(dim=(-2, -1))              # torch.Size([2, 128])
        prompt_weights = F.softmax(self.linear_proj(emb), dim=1)        # torch.Size([2, 5])

        return prompt_weights
    
class FlowPromptInteraction(nn.Module):
    """
        Generating an input-conditioned prompt and integrate it into features.
    """
    def __init__(self, embed_dim=64, prompt_dim=96, prompt_len=5, prompt_size=96, num_blocks=3, align=False):
        super(FlowPromptInteraction, self).__init__()
        self.align = align
        self.prompt_dim = prompt_dim

        self.prompt_param_stc = nn.Parameter(torch.rand(prompt_len, prompt_dim, prompt_size, prompt_size))      # torch.Size([5, 96, 96, 96])
        self.prompt_param_dyn = nn.Parameter(torch.rand(prompt_len, prompt_dim, prompt_size, prompt_size))      # torch.Size([5, 96, 96, 96])
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

        # self.residual = ResidualBlocksWithInputConv(embed_dim + prompt_dim, embed_dim, num_blocks)
        self.residual = nn.Sequential(
            nn.Conv2d(embed_dim + prompt_dim, embed_dim, 3, 1, 1),
            NAFBlock(embed_dim)
            )

    def forward(self, prompt_stc_weights, prompt_dyn_weights, x):
        b, c, h, w = x.shape                    # torch.Size([2, 96, 64, 64])

        input_conditioned_prompt_stc = prompt_stc_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param_stc.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # torch.Size([2, 5, 96, 96, 96])
        input_conditioned_prompt_dyn = prompt_dyn_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param_dyn.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # torch.Size([2, 5, 96, 96, 96])
        input_conditioned_prompt = input_conditioned_prompt_dyn * input_conditioned_prompt_stc
        input_conditioned_prompt = torch.sum(input_conditioned_prompt, dim=1)                           # torch.Size([2, 96, 96, 96])
        input_conditioned_prompt = F.interpolate(input_conditioned_prompt, (h, w), mode="bilinear")     # torch.Size([2, 96, 64, 64])
        input_conditioned_prompt = self.conv(input_conditioned_prompt) # b, promptdim, h, w             # torch.Size([2, 96, 64, 64])

        output = self.residual(torch.cat([x, input_conditioned_prompt], dim=1))                         # torch.Size([2, 96, 64, 64])

        return output

@ARCH_REGISTRY.register()
class ORCANet(nn.Module):
    def __init__(self,
                 mid_channels=64,
                 num_blocks=5,
                 max_residue_magnitude=10,
                 spynet_pretrained=None,
                 keyframe_interval=6,
                 prompt_size=96,
                 prompt_dim=96,
                 cpu_cache_length=40):

        super().__init__()
        self.mid_channels = mid_channels
        self.prompt_dim = prompt_dim
        self.prompt_size = prompt_size
        self.cpu_cache_length = cpu_cache_length
        self.keyframe_interval = keyframe_interval
        enc_blk_nums = [1,6]
        middle_blk_num = 1
        dec_blk_nums = [1,1]
        chan = mid_channels
        # depth estimate
        model_configs = {
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
            }

        # optical flow
        self.spynet = SpyNet(load_path=spynet_pretrained)
        self.dehazeinit = CIED(self.mid_channels, min_beta=0.001, max_beta=0.0, min_d=0.2, max_d=5)

        encoder = 'vits'
        config = model_configs[encoder]
        self.depth = DepthAnything(config)
        # self.depth = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').eval().cuda()
        self.depth.load_state_dict(torch.load(f'experiments/pretrained_models/preDepthanything_ckpt/depth_anything_{encoder}14.pth'))
        # self.depth = self.depth.to('cuda').eval()

        # shallow feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 1, 1),
            NAFBlock(chan),
            nn.Conv2d(chan, 2*chan, 2, 2),
            NAFBlock(2*chan),
            nn.Conv2d(2*chan, 4*chan, 2, 2),
            )
        
        self.PGM_static = FlowPromptGenerationModule(embed_dim=4*mid_channels, prompt_dim=self.prompt_dim,
                                                    prompt_size=self.prompt_size)
        
        self.PGM_dyn = FlowPromptGenerationModule(embed_dim=4*mid_channels, prompt_dim=self.prompt_dim,
                                                    prompt_size=self.prompt_size)
        
        self.FPI = FlowPromptInteraction(embed_dim=4*mid_channels, prompt_dim=self.prompt_dim,
                                                    prompt_size=self.prompt_size)
        
        self.linear_proj = nn.Linear(4*mid_channels, 5)
        self.phi = nn.Sigmoid()
        self.key_fusion = nn.Conv2d(8*self.mid_channels, 4*self.mid_channels, 3, 1, 1, bias=True)

            
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.middle_blks = nn.ModuleDict()
        self.ups = nn.ModuleDict()
        self.downs = nn.ModuleDict()
        self.mitigate = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            chan = 4*mid_channels
            self.encoders[module] = nn.ModuleList()
            self.decoders[module] = nn.ModuleList()
            self.middle_blks[module] = nn.ModuleList()
            self.ups[module] = nn.ModuleList()
            self.downs[module] = nn.ModuleList()

            for num in enc_blk_nums:
                self.encoders[module].append(
                    nn.Sequential(
                        *[NAFBlock(chan) for _ in range(num)],
                    )
                )
                self.downs[module].append(
                    nn.Conv2d(chan, 2*chan, 2, 2)
                )
                chan = chan * 2         # 64 128 256 512

            self.middle_blks[module] = \
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(middle_blk_num)]
                )
            
            for num in dec_blk_nums:
                self.ups[module].append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
                chan = chan // 2
                self.decoders[module].append(
                    nn.Sequential(
                        *[NAFBlock(chan) for _ in range(num)]
                    )
                )
            self.mitigate[module] = nn.Sequential(
                nn.Conv2d((2 + i) *4*mid_channels, 4*mid_channels, 1, bias=False),
                NAFBlock(4*mid_channels)
            )
            # self.mitigate[module] = ResidualBlocksWithInputConv(
            #     (2 + i) *4*mid_channels, 4*mid_channels, num_blocks)

        for num in dec_blk_nums:
            self.ups[module].append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders[module].append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
        self.findec = nn.Sequential(
            nn.Conv2d(chan, chan * 2, 1, bias=False),  # 256 512
            nn.PixelShuffle(2),
            NAFBlock(chan//2),
            nn.Conv2d(chan//2, chan, 1, bias=False),   # 256 128 64
            nn.PixelShuffle(2),
            NAFBlock(chan//4),                         # 64 
            )
        
        # self.reconstruction = ResidualBlocksWithInputConv(
        #     3 * mid_channels, mid_channels, num_blocks)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(5*4*mid_channels, 8*mid_channels, 1, bias=False),
            nn.PixelShuffle(2),
            NAFBlock(2*mid_channels),
            nn.Conv2d(2*mid_channels, 4*mid_channels, 1, bias=False),
            nn.PixelShuffle(2),
            NAFBlock(mid_channels),
        )
            
        self.ending = nn.Conv2d(in_channels=mid_channels, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
            bias=True)

        self.padder_size = 2 ** len(self.encoders)

        ##                           064                     loss_prompt                              loss_prompt                  
        ##  input: depth, lq img     128        enc2 <--- prompt = pf * pb   dec3        enc2 <--- prompt             dec3
        ##      BetaEstimator        256    flow---> enc3               dec2   --->  flow---> enc3               dec2
        ##                           512                       mid                                      mid          
        ##  out: initframe, beta    1024                                                                         
        ##                  loss_beta    


    def getStaticPrompts(self, x, keyframe_idx):
        prompts_static = {}
        for i in keyframe_idx:
            if self.cpu_cache:
                x_i = x[i].cuda()
            else:
                x_i = x[i]                      # torch.Size([2, 96, 64, 64])

            prompts_static[i] = self.PGM_static(x_i)      # torch.Size([2, 96, 64, 64])

            if self.cpu_cache:
                prompts_static[i] = prompts_static[i].cpu()
                torch.cuda.empty_cache()
        return prompts_static

    def compute_flow(self, lqs):

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def finaldec(self, lqs, feats):
        outputs = []                                                    # lqs: torch.Size([2, 12, 3, 256, 256])
        num_outputs = len(feats['spatial'])     # 12

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial' and k != 'beta']     # torch.Size([2, 96, 64, 64]) feats[k].pop(0) derive the first frame and delete it
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)                                   # torch.Size([2, 256*5, 64, 64]) 
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)                            # torch.Size([2, 96, 64, 64])
            hr = self.ending(hr)                                 # torch.Size([2, 3, 256, 256])

            hr += lqs[:, i, :, :, :].cuda()                         # torch.Size([2, 3, 256, 256])

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)
    
    def closest_multiple_of_14(self,n):
        return round(n / 14.0) * 14

    def forward(self, lqs):
        n, t, c, h0, w0 = lqs.size()                              # torch.Size([2, 12, 3, 256, 256])
        # if not self.training:
        #     self.spynet.to('cpu')

        # whether to cache the features in CPU (no effect if using CPU)
        if t >= self.cpu_cache_length:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            # feats['depth'] = []
            feats['beta'] = []
            lqstrs = []
            for i in range(0, t):
                framei = lqs[:, i, :, :, :].cuda()              # 1, 3, 64, 64

                target_height = self.closest_multiple_of_14(framei.shape[2])
                target_width = self.closest_multiple_of_14(framei.shape[3])  # 1, 3, 540, 960
                depth, _ = self.depth(tfs.Resize([target_height, target_width])(framei))
                depth = F.interpolate(depth.unsqueeze(1), (h0, w0), mode='bilinear', align_corners=False)
                # feats['depth'].append(depth.cpu())
                
                framei, betai = self.dehazeinit(framei, depth, require_paras=True)
                feat = self.feat_extract(framei).cpu()
                lqstrs.append(framei.cpu())
                feats['spatial'].append(feat)
                # feats['beta'].append(betai)

                torch.cuda.empty_cache()
            h, w = feat.shape[2:]
            lqstrs = torch.stack(lqstrs, dim=1).cuda()
            lqs = lqs.cpu()
            torch.cuda.empty_cache()
        else:
            lqstrs = lqs.view(-1, c, h0, w0)
            target_height = self.closest_multiple_of_14(lqs.shape[-2])
            target_width = self.closest_multiple_of_14(lqs.shape[-1])  # 1, 3, 540, 960
            depth, _ = self.depth(tfs.Resize([target_height, target_width])(lqstrs))
            depth = F.interpolate(depth.unsqueeze(1), (h0, w0), mode='bilinear', align_corners=False)

            lqstrs, betai = self.dehazeinit(lqstrs, depth, require_paras=True)        # [24 3 256 256]  [24 1 1 1]
            feats_ = self.feat_extract(lqstrs)   # torch.Size([24, 128, 64, 64]) shallow_feature
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)                # torch.Size([2, 12, 256, 64, 64])
            lqstrs = lqstrs.view(n, t, -1, h*4, w*4)
            depth = depth.view(n, t, -1, h*4, w*4)                # torch.Size([2, 12, 256, 64, 64])
            betai = betai.view(n, t, -1, 1, 1)                  # [2, 12, 1, 1, 1]
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
            # feats['depth'] = [depth[:, i, :, :, :] for i in range(0, t)]
            feats['beta'] = [betai[:, i, :, :, :] for i in range(0, t)]
            beta_pred = torch.cat([b.view(-1) for b in feats['beta']])
            

        lqs_downsample = F.interpolate(
            lqstrs.view(-1, c, h*4, w*4), scale_factor=0.25,
            mode='bilinear').view(n, t, c, h, w)           # torch.Size([2, 12, 3, 64, 64])        

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
            
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)   # torch.Size([2, 11, 2, 64, 64])

        # generate keyframe features
        keyframe_idx = list(range(0, t, self.keyframe_interval))
        if keyframe_idx[-1] != t - 1:
            keyframe_idx.append(t - 1)  # last frame is a keyframe
        prompts_static = self.getStaticPrompts(feats['spatial'], keyframe_idx)

        # feature propagation
        for iter_ in [1,2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward                                  # torch.Size([2, 11, 2, 64, 64])
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                n, t, _, h, w = flows.size()
                frame_idx = list(range(0, t + 1))
                flow_idx = list(range(-1, t))

                if direction == 'backward':
                    frame_idx = frame_idx[::-1]     # [11,10,9,...,1,0]
                    flow_idx = frame_idx            # [11,10,9,...,1,0]

                feat_prop = flows.new_zeros(n, self.mid_channels, h, w).cuda()     # torch.Size([2, 64, 64, 64])
                q_i_1 = 0

                for i, idx in enumerate(frame_idx):  # i=0 idx=0 i=12,idx=12
                    x_i = feats['spatial'][idx]                    # torch.Size([1, 128, 64, 64]) shallow_feature
                    if self.cpu_cache:
                        x_i = x_i.cuda()
                        feat_prop = feat_prop.cuda()

                    # x_i = self.PGI_prop(x_i)
                    q_i = self.PGM_dyn(x_i)

                    if i > 0:
                        flow = flows[:, flow_idx[i], :, :, :].cuda()               # torch.Size([2, 2, 64, 64])
                        if self.cpu_cache:
                            flow = flow.cuda()
                        feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  # torch.Size([2, 96, 64, 64]) warped feature
                        x_i = self.key_fusion(torch.cat([x_i, feat_prop], dim=1))

                    if idx in keyframe_idx:
                        if self.cpu_cache:
                            k_s = prompts_static[idx].cuda()
                        else:
                            k_s = prompts_static[idx]          # torch.Size([2, 96, 64, 64])

                    gamma_i = self.phi(self.linear_proj(x_i.mean(dim=(-2, -1))))
                    q_i = gamma_i * q_i + q_i_1
                    q_i_1 = q_i
                    x_i = self.FPI(k_s, q_i, x_i)

                    encs = []
                    for encoder, down, num in zip(self.encoders[module], self.downs[module], [1,2,3,4]):
                        x_i = encoder(x_i)      # [1 64 h w] -> [1 128 h/2 h/2] -> [1 256 h/4 h/4] -> [1 512 h/8 h/8]  -> [1 1024 h/16 h/16]
                        encs.append(x_i)
                        x_i = down(x_i)

                    x_i = self.middle_blks[module](x_i)

                    for decoder, up, enc_skip, num in zip(self.decoders[module], self.ups[module], encs[::-1], [1,2,3,4]):
                        x_i = up(x_i)
                        x_i = x_i + enc_skip
                        x_i = decoder(x_i)

                    # concatenate the residual info
                    feat = [feats['spatial'][idx]] + [                                                                    # torch.Size([2, 192, 64, 64])
                        feats[k][idx]
                        for k in feats if k not in ['spatial', 'beta', module]
                    ] + [x_i]
                    if self.cpu_cache:
                        feat = [f.cuda() for f in feat]

                    feat = torch.cat(feat, dim=1)                           # cat(x_i, feat_prop) torch.Size([2, 64*(2+i), 64, 64]) 192/288
                    x_i = self.mitigate[module](feat)     # torch.Size([2, 256, 64, 64])
                    if self.cpu_cache:
                        x_i = x_i.cpu()
                        torch.cuda.empty_cache()
                    feats[module].append(x_i)
                    feat_prop = x_i

                    del encs
                if direction == 'backward':
                    feats[module] = feats[module][::-1]

                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()
        if self.training:
            return self.finaldec(lqstrs, feats)[:,:,:,:h0,:w0], beta_pred, prompts_static
        else:
            return self.finaldec(lqstrs, feats)[:,:,:,:h0,:w0]


class TransmissionEstimator(nn.Module):
    def __init__(self, width=15,):
        super(TransmissionEstimator, self).__init__()
        self.width = width
        self.t_min = 0.2
        self.alpha = 2.5
        self.A_max = 220.0/255
        self.omega=0.95
        self.p = 0.001
        self.max_pool = nn.MaxPool2d(kernel_size=width,stride=1)
        self.max_pool_with_index = nn.MaxPool2d(kernel_size=width, return_indices=True)
        # self.guided_filter = GuidedFilter(r=40,eps=1e-3)

    def get_dark_channel(self, x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.width//2, self.width//2,self.width//2, self.width//2), mode='constant', value=1)
        x = -(self.max_pool(-x))
        return x

    def get_atmosphere_light(self,I,dc):
        n,c,h,w = I.shape
        flat_I = I.view(n,c,-1)
        flat_dc = dc.view(n,1,-1)
        searchidx = torch.argsort(flat_dc, dim=2, descending=True)[:,:,:int(h*w*self.p)]
        searchidx = searchidx.expand(-1,3,-1)
        searched = torch.gather(flat_I,dim=2, index=searchidx)
        return torch.max(searched, dim=2 ,keepdim=True)[0].unsqueeze(3)

    def get_transmission(self, I, A):
        return 1-self.omega* self.get_dark_channel(I/A)

    # def get_refined_transmission(self, I, rawt):
    #     I_max = torch.max(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    #     I_min = torch.min(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    #     normI = (I - I_min)/(I_max-I_min)
    #     refinedT = self.guided_filter(normI, rawt)

    #     return refinedT

    def get_radiance(self,I, A, t):
        return (I-A)/t + A

    def get_depth(self, I):
        I_dark = self.get_dark_channel(I)

        A = self.get_atmosphere_light(I, I_dark)
        A[A>self.A_max] = self.A_max
        rawT = self.get_transmission(I, A)

        # print(I)

        refinedT = self.get_refined_transmission(I, rawT)
        return refinedT

    def get_atmosphere_light_new(self, I):
        I_dark = self.get_dark_channel(I)
        A = self.get_atmosphere_light(I, I_dark)
        A[A > self.A_max] = self.A_max
        return A

class CIED(nn.Module):
    def __init__(self, base_channel_nums, init_weights=True, min_beta=0.001, max_beta=0.1, min_d=0.3, max_d=5):
        super(CIED, self).__init__()
        self.transmission_estimator = TransmissionEstimator()
        self.use_dc_A = False
        exportable = True
        self.MIN_BETA=min_beta
        self.MAX_BETA=max_beta
        self.MIN_D = min_d
        self.MAX_D = max_d

        # use_pretrained = False if os.path.exists(path) else True
        self.pretrained =_make_pretrained_efficientnet_lite3(False, exportable=exportable)
        self.groups = 1

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=[1,1])
        self.final_conv_beta_1 = nn.Conv2d(
            in_channels=32 +48 +136 +384,
            out_channels=2*base_channel_nums,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True
        )
        self.final_conv_beta_2 = nn.Conv2d(
            in_channels=2*base_channel_nums,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True
        )

        if init_weights:
            self.init_weights('xaiver')

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            # print(m.name)
            # if classname.find('pretrained') != -1:
            #     print(classname)
            #     return
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


    def forward_get_A(self, x): # output A: N,3,1,1
        if self.use_dc_A:
            A = self.transmission_estimator.get_atmosphere_light_new(x)
        else:
            A = x.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)

        return A

    def forward(self, x_0, depth, require_paras=True, use_guided_filter=False):

        layer_1 = self.pretrained.layer1(x_0)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_beta = F.adaptive_avg_pool2d(layer_1, [1, 1]).detach()
        layer_2_beta = F.adaptive_avg_pool2d(layer_2, [1, 1]).detach()
        layer_3_beta = F.adaptive_avg_pool2d(layer_3, [1, 1]).detach()
        layer_4_beta = F.adaptive_avg_pool2d(layer_4, [1, 1]).detach()

        beta = self.final_conv_beta_1(torch.cat([layer_1_beta, layer_2_beta, layer_3_beta, layer_4_beta], dim=1))
        beta = self.final_conv_beta_2(beta)

        beta = self.MIN_BETA + (self.MAX_BETA-self.MIN_BETA)*(torch.tanh(beta) + 1) / 2

        t = torch.exp(-beta * depth)  # t(z) = e^{-β d(z)}
        t = t.clamp(0.05, 0.95)       # 避免极端值

        if use_guided_filter:
            t = self.transmission_estimator.get_refined_transmission(x_0,t)       # 清晰图 J、大气光 A、深度 d、雾浓度 β、透射率 t。

        A = self.forward_get_A(x_0)

        if require_paras:
            return ((x_0-A)/t + A).clamp(0,1), beta
        else:
            return ((x_0-A)/t + A).clamp(0,1), t

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
