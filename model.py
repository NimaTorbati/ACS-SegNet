import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import matplotlib.pyplot as plt
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBlock11(nn.Module):
    """
    Fuse two encoder feature maps using a lightweight ConvAdapter module
    instead of direct concatenation.

    Args:
        in_channels1: number of channels in encoder1 feature map
        in_channels2: number of channels in encoder2 feature map
        out_channels: number of output channels for the fused feature
        reduction: reduction ratio for bottleneck (e.g., 4 → bottleneck_channels = out_channels // 4)
        mode: fusion mode - ['add', 'concat', 'gate']
    """

    def __init__(self, in_channels1, in_channels2, out_channels, reduction=4, mode='concat'):
        super().__init__()
        self.mode = mode

        # Align both encoder feature maps to a shared dimension
        self.proj1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1, bias=False)

        # Bottleneck (ConvAdapter)
        bottleneck_channels = max(out_channels // reduction, 4)
        self.adapter = nn.Sequential(
            nn.Conv2d(2*out_channels, bottleneck_channels, kernel_size=3, padding=1, groups=bottleneck_channels,
                      bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Gating parameter (optional, learnable α)
        self.alpha = nn.Parameter(torch.ones(1))
        self.att = FusionBlock9(channels = in_channels1 + in_channels2 + out_channels)
        # Optional gating mechanism for 'gate' mode
        if mode == 'gate':
            self.gate = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, feat1, feat2):
        # Ensure same spatial size (ViT might be smaller)
        if feat1.size()[2:] != feat2.size()[2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)

        f1 = self.proj1(feat1)
        f2 = self.proj2(feat2)

        if self.mode == 'concat':
            fused = torch.cat([f1, f2], dim=1)
            fused = self.adapter(fused)
            fused = torch.cat([fused], dim=1)
            # fused = torch.cat([feat1, feat2, fused], dim=1)
            # fused = self.att(fused)
        elif self.mode == 'gate':
            gate = self.gate(torch.cat([f1, f2], dim=1))
            fused = f1 * gate + f2 * (1 - gate)
            fused = fused + self.alpha * self.adapter(fused)
        else:  # default: 'add'
            fused = (f1 + f2) / 2
            fused = fused + self.alpha * self.adapter(fused)

        return fused


class FusionBlock1(nn.Module):
    def __init__(self, u_ch, s_ch, wa_out, wb_out):
        super().__init__()
        self.s_proj = nn.Conv2d(s_ch, u_ch, kernel_size=1)
        self.wa = nn.Conv2d(u_ch, wa_out, kernel_size=3, padding=1)
        self.wb = nn.Conv2d(u_ch, wb_out, kernel_size=3, padding=1)
        self.att = FusionBlock9(channels=wb_out)
    def forward(self, U, S):
        P = self.s_proj(S)
        M_cos = U * torch.cos(P)
        M_sin = U * torch.sin(P)
        fused = self.wa(M_cos) - self.wb(M_sin)
        # fused = self.att(fused) + fused
        return fused

class FusionBlock6(nn.Module):
    def __init__(self, u_ch, s_ch, wa_out, wb_out):
        super().__init__()
        self.s_proj = nn.Conv2d(s_ch, u_ch, kernel_size=1)
        self.s_proj1 = nn.Conv2d(u_ch, s_ch, kernel_size=1)

        self.wa = nn.Conv2d(u_ch, wa_out, kernel_size=3, padding=1)
        self.wb = nn.Conv2d(u_ch, wb_out, kernel_size=3, padding=1)

        self.wc = nn.Conv2d(s_ch, wa_out, kernel_size=3, padding=1)
        self.wd = nn.Conv2d(s_ch, wb_out, kernel_size=3, padding=1)


    def forward(self, U, S):
        P = self.s_proj(S)
        M_cos = U * torch.cos(P)
        M_sin = U * torch.sin(P)
        fused = self.wa(M_cos) - self.wb(M_sin)


        P = self.s_proj1(U)
        M_cos = S * torch.cos(P)
        M_sin = S * torch.sin(P)
        fused = torch.cat([self.wc(M_cos) - self.wd(M_sin), fused], dim=1)
        return fused



class FusionBlock5(nn.Module):
    def __init__(self, u_ch, s_ch, wa_out, mid_size):
        """
        u_ch: number of channels in U (local features)
        s_ch: number of channels in S (spatial features)
        mid_dim: dimension for bilinear interaction space
        """
        super().__init__()

        # Direct learnable projections for channels
        mid_size = int(mid_size/2)
        self.wa = nn.Parameter(torch.randn(s_ch, mid_size))
        self.wb = nn.Parameter(torch.randn(u_ch, mid_size))

        # Output projection to match U's channel count
        self.out_proj = nn.Conv2d(wa_out, u_ch, kernel_size=1)
        self.mid_size = mid_size

    def forward(self, U, S):
        """
        U: (B, u_ch, H, W) - local features
        S: (B, s_ch, H, W) - spatial features
        """
        B, _, H, W = U.shape
        N = H * W

        # Project without flattening
        S_proj = torch.einsum('bchw,cm->bmhw', S, self.wa)  # (B, mid_dim, H, W)
        U_proj = torch.einsum('bchw,cm->bmhw', U, self.wb)  # (B, mid_dim, H, W)

        # Flatten *after* projection
        S_proj_flat = S_proj.view(B, S_proj.shape[1], -1)  # (B, mid_dim, N)
        U_proj_flat = U_proj.view(B, U_proj.shape[1], -1)  # (B, mid_dim, N)

        attn = torch.bmm(S_proj_flat, U_proj_flat.transpose(1, 2))  # (B, mid_dim, mid_dim)
        # (B, mid_dim, mid_dim)
        attn = attn.view(B, 1, self.mid_size, self.mid_size)  # still 2D
        attn = F.interpolate(attn, size=U.shape[2:], mode="bilinear", align_corners=False)

        # Reshape to spatial map
        # attn = attn.view(B,1, H, W)
        fused = attn * torch.cat([U, S], dim=1) + torch.cat([U, S], dim=1)  # Concatenate along channel dimension
        return fused


class FusionBlock2(nn.Module):
    def __init__(self, u_ch, s_ch, wa_out, wb_out):
        super().__init__()
        self.s_proj = nn.Conv2d(s_ch, u_ch, kernel_size=1)
        self.wa = nn.Conv2d(u_ch, wa_out, kernel_size=3, padding=1)
        self.wb = nn.Conv2d(u_ch, wb_out, kernel_size=3, padding=1)

    def forward(self, U, S):
        P = self.s_proj(S)
        M_cos = U * torch.cos(P)
        M_sin = U * torch.sin(P)
        fused = torch.cat([U, S, self.wa(M_cos) - self.wb(M_sin)], dim=1)

        return fused


class FusionBlock3(nn.Module):
    def __init__(self, u_ch, s_ch, wa_out, wb_out):
        super().__init__()
        self.s_proj = nn.Conv2d(s_ch, u_ch, kernel_size=1)
        self.wa = nn.Conv2d(u_ch, wa_out, kernel_size=3, padding=1)
        self.wb = nn.Conv2d(u_ch, wb_out, kernel_size=3, padding=1)
        self.att = FusionBlock9(channels=u_ch + s_ch + wa_out)
    def forward(self, U, S):
        P = self.s_proj(S)
        M_cos = U * torch.cos(P)
        M_sin = U * torch.sin(P)
        fused = torch.cat([U,S, self.wa(M_cos) - self.wb(M_sin)], dim=1)
        fused = self.att(fused)


        return fused


class FusionBlock4(nn.Module):
    def __init__(self, u_ch, s_ch, wa_out, wb_out):
        super().__init__()

    def forward(self, U, S):
        fused = U#
        return fused




class FusionBlock0(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, U, S):
        # Resize S if needed to match U
        if S.shape[2:] != U.shape[2:]:
            S = F.interpolate(S, size=U.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([U, S], dim=1)  # Concatenate along channel dimension
        return fused






class FusionBlock7(nn.Module):
    """
    Multi-scale attention gate
    """
    def __init__(self, channel):
        super(FusionBlock7, self).__init__()
        self.channel = channel
        self.Conv1 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(inplace=True)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(inplace=True)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, stride=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.sig=nn.Sigmoid()

    def forward(self, U, S):
        if S is None:
            x = U
        else:
            x = torch.cat([U, S], dim=1)
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        avg_x1=torch.mean(x1,dim=1,keepdim=True)
        max_x2=torch.max(x2,dim=1,keepdim=True).values
        x3=self.Conv3(torch.cat((avg_x1,max_x2),dim=1))
        x4=self.sig(x3)
        return x+x*x4



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale

#
class FusionBlock9(nn.Module):
    """
    CBAM with residual connection:
    out = x + CBAM(x)
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(FusionBlock9, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, U, S = None):
        if S is None:
            x = U
        else:
            x = torch.cat([U, S], dim=1)
        att = self.channel_att(x)
        att = self.spatial_att(att)

        return att   # residual addition




class FusionBlock10(nn.Module):
    """
    CBAM with residual connection:
    out = x + CBAM(x)
    """
    def __init__(self, channels_u,channels_s, reduction=16, spatial_kernel=7):
        super(FusionBlock10, self).__init__()
        self.channel_att_u = ChannelAttention(channels_u, reduction)
        self.channel_att_s = ChannelAttention(channels_s, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, U, S):
        # if S is None:
        #     x = U
        # else:
        #     x = torch.cat([U, S], dim=1)
        att_u = self.channel_att_u(U)
        att_u = self.spatial_att(att_u)

        att_s = self.channel_att_s(S)
        att_s = self.spatial_att(att_s)

        U = att_u + U
        S = att_s + S



        return torch.cat([U, S], dim=1)   # residual addition




class FusionBlock8(nn.Module):
    def __init__(self, u_ch, s_ch, wa_out, wb_out):
        super().__init__()
        self.s_proj = nn.Conv2d(s_ch, u_ch, kernel_size=1)
        self.s_proj1 = nn.Conv2d(u_ch, s_ch, kernel_size=1)

        self.wa = nn.Conv2d(u_ch, wa_out, kernel_size=3, padding=1)
        self.wb = nn.Conv2d(u_ch, wb_out, kernel_size=3, padding=1)

        self.wc = nn.Conv2d(s_ch, wa_out, kernel_size=3, padding=1)
        self.wd = nn.Conv2d(s_ch, wb_out, kernel_size=3, padding=1)
        self.att = FusionBlock7(channel=2*wa_out)

    def forward(self, U, S):
        P = self.s_proj(S)
        M_cos = U * torch.cos(P)
        M_sin = U * torch.sin(P)
        fused = self.wa(M_cos) - self.wb(M_sin)
        # fused = self.att(fused,None) + fused

        P = self.s_proj1(U)
        M_cos = S * torch.cos(P)
        M_sin = S * torch.sin(P)
        fused = torch.cat([self.wc(M_cos) - self.wd(M_sin), fused], dim=1)
        fused = self.att(fused,None) + fused
        return fused


class DualEncoderUNet(nn.Module):
    def __init__(
        self,
        m=128,
        unet_encoder_name="resnet34",
        unet_encoder_weights=None,
        segformer_variant="nvidia/segformer-b2-finetuned-ade-512-512",
        wa_outs=None,
        wb_outs=None,
        classes=1,
        decoder_channels=(256, 128, 64, 32,16),
        simple_fusion=0,
        regression=False,
        in_channels=3,
        freeze_segformer = False,
        freeze_unet=False,
        input_size=1024,
        decoder_type="unet",
        IgnoreBottleNeck = False,
            cof_seg = 1,
            cof_unet = 1,
            model_depth=5,
    ):
        super().__init__()
        self.classes = classes
        self.cof_seg = cof_seg
        self.cof_unet = cof_unet
        self.freeze_segformer = freeze_segformer
        self.freeze_unet = freeze_unet
        self.unet_encoder = smp.encoders.get_encoder(
            unet_encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=unet_encoder_weights,
        )
        u_out_channels = self.unet_encoder.out_channels[1:]  # [64, 64, 128, 256, 512]
        self.IgnoreBottleNeck = IgnoreBottleNeck
        seg_cfg = SegformerConfig.from_pretrained(segformer_variant)
        seg_cfg.output_hidden_states = True
        self.segformer = SegformerModel.from_pretrained(segformer_variant, config=seg_cfg)
        self.register_buffer('segformer_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('segformer_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if wa_outs is None:
            wa_outs = [m] * 4
        if wb_outs is None:
            wb_outs = [m] * 4
        assert len(wa_outs) == 4 and len(wb_outs) == 4
        wa_outs = [2*s for s in u_out_channels[1:]]
        wb_outs = [2*s for s in u_out_channels[1:]]
        assert len(wa_outs) == 4 and len(wb_outs) == 4
        s_expected = list(seg_cfg.hidden_sizes[-4:])
        # Fusion blocks for first 4 skips (index 0 to 3)
        self.fusions = nn.ModuleList()
        # mid_size = int(input_size/2)
        for i in range(4):  # i = 0,1,2,3 for skips 0..3
            u_ch = u_out_channels[i]
            s_ch = s_expected[i]
            wa_out = wa_outs[i]
            wb_out = wb_outs[i]
            if simple_fusion == 0:
                self.fusions.append(FusionBlock0())

            elif simple_fusion == 1:
                self.fusions.append(FusionBlock1(u_ch=u_ch, s_ch=s_ch, wa_out=wa_out, wb_out=wb_out))
            elif simple_fusion == 2:
                self.fusions.append(FusionBlock2(u_ch=u_ch, s_ch=s_ch, wa_out=wa_out, wb_out=wb_out))
            elif simple_fusion == 3:
                self.fusions.append(FusionBlock3(u_ch=u_ch, s_ch=s_ch, wa_out=wa_out, wb_out=wb_out))
            elif simple_fusion == 4:
                self.fusions.append(FusionBlock4(u_ch=u_ch, s_ch=s_ch, wa_out=wa_out, wb_out=wb_out))
            elif simple_fusion == 5:
                self.fusions.append(FusionBlock5(u_ch=u_ch, s_ch=s_ch, wa_out=wa_out, mid_size=mid_size))
                mid_size = int(mid_size / 2)
            if simple_fusion == 6:
                self.fusions.append(FusionBlock6(u_ch=u_ch, s_ch=s_ch, wa_out=wa_out, wb_out=wb_out))
            if simple_fusion == 7:
                self.fusions.append(FusionBlock7(channel = u_ch + s_ch))
            if simple_fusion == 8:
                self.fusions.append(FusionBlock8(u_ch=u_ch, s_ch=s_ch, wa_out=wa_out, wb_out=wb_out))

            if simple_fusion == 9:
                self.fusions.append(FusionBlock9(channels = u_ch + s_ch))

            if simple_fusion == 10:
                self.fusions.append(FusionBlock10(channels_u = u_ch, channels_s =s_ch))

            if simple_fusion == 11:
                self.fusions.append(FusionBlock11(in_channels1 = u_ch, in_channels2 = s_ch, out_channels= u_ch))

        if simple_fusion == 0:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = s_expected[i] + u_out_channels[i]
        elif simple_fusion == 1:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
        elif simple_fusion == 2:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = encoder_channels_for_decoder[i] + s_expected[i] + u_out_channels[i]
        elif simple_fusion == 3:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = encoder_channels_for_decoder[i] + u_out_channels[i] + s_expected[i]
        elif simple_fusion == 4:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = u_out_channels[i]

        elif simple_fusion == 5:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = s_expected[i] + u_out_channels[i]

        elif simple_fusion == 6:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = 2 * wa_outs[i]
        elif simple_fusion == 7:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = s_expected[i] + u_out_channels[i]

        elif simple_fusion == 8:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = 2 * wa_outs[i]
        elif simple_fusion == 9:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                encoder_channels_for_decoder[i] = s_expected[i] + u_out_channels[i]
        elif simple_fusion == 11:
            encoder_channels_for_decoder = wa_outs + [u_out_channels[4]]
            for i in range(4):
                # encoder_channels_for_decoder[i] = s_expected[i] + 2*u_out_channels[i]
                encoder_channels_for_decoder[i] = u_out_channels[i]
        # Decoder expects 5 skips: 4 fused + 1 bottleneck (last unet encoder output)
        # The bottleneck (last skip) is u_out_channels[4] and added without fusion
        # ----- choose decoder type -----
        if decoder_type.lower() == "unet":
            DecoderClass = UnetDecoder
        elif decoder_type.lower() in ("unet++", "unetplusplus"):
            DecoderClass = UnetPlusPlusDecoder
        else:
            raise ValueError(f"Unknown decoder_type '{decoder_type}'. Use 'unet' or 'unet++'.")

        if self.IgnoreBottleNeck:
            encoder_channels_for_decoder = encoder_channels_for_decoder[:-1] + [0]  # keep first 4 (skip0..skip3)
            decoder_channels = decoder_channels
            print('skip')
            n_blocks = 5
        else:
            n_blocks = 5

        self.decoder = DecoderClass(
            encoder_channels=[in_channels] + encoder_channels_for_decoder,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
            use_batchnorm=True,
            IgnoreBottleNeck=self.IgnoreBottleNeck
        )

        if regression:
            # Regression: keep activation if you want non-negative outputs
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(decoder_channels[-1], 1, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            # Classification / segmentation
            self.segmentation_head = nn.Conv2d(
                decoder_channels[-1],
                self.classes,
                kernel_size=3,
                padding=1
            )
            # No activation here — leave logits for the loss function

    def _filter_and_sort_unet_feats(self, u_feats, input_h):
        filtered = [f for f in u_feats if f.shape[2] < input_h]
        filtered = sorted(filtered, key=lambda t: t.shape[2], reverse=True)
        return filtered

    def _sort_segf_feats(self, s_feats):
        return sorted(s_feats, key=lambda t: t.shape[2], reverse=True)

    def forward(self, x, debug_print_shapes=False):
        cof_seg = self.cof_seg

        cof_unet = self.cof_unet

        B, C, H_in, W_in = x.shape

        # Normalize first 3 channels for segformer input
        # Handle possible extra channels (like depth or others)
        x_for_segformer = cof_seg*x[:, :3, :, :]
        x_for_segformer = (x_for_segformer - self.segformer_mean) / self.segformer_std

        if C > 3:
            # concat back extra channels (un-normalized)
            x_for_segformer = torch.cat([x_for_segformer, x[:, 3:, :, :]], dim=1)

        # Proceed with encoders
        u_feats_all = self.unet_encoder(cof_unet*x)

        u_feats = self._filter_and_sort_unet_feats(u_feats_all, H_in)


        s_all = self.segformer(pixel_values=x_for_segformer).hidden_states
        s_feats = s_all[-4:]
        s_feats = sorted(s_feats, key=lambda t: t.shape[2], reverse=True)




        if debug_print_shapes:
            print("== Encoder shapes after filtering & sorting ==")
            for i, f in enumerate(u_feats, start=1):
                print(f"U{i}: {tuple(f.shape)}")
            for i, f in enumerate(s_feats, start=1):
                print(f"S{i}: {tuple(f.shape)}")

        skips = []
        skips.append(torch.zeros_like(x))

        # Fuse SegFormer with first 4 U-Net skips
        for i in range(4):
            U = u_feats[i]
            S = s_feats[i]
            if (S.shape[2] != U.shape[2]) or (S.shape[3] != U.shape[3]):
                S = F.interpolate(S, size=(U.shape[2], U.shape[3]), mode="bilinear", align_corners=False)
            if debug_print_shapes:
                print(f"Fusing skip {i} shapes: U{tuple(U.shape)} S(resized) {tuple(S.shape)}")
            fused = self.fusions[i](U, S)
            skips.append(fused)

        # Add bottleneck skip (last U-Net encoder output) without fusion
        bottleneck = cof_unet*u_feats_all[-1]
        if self.IgnoreBottleNeck:
            bottleneck = torch.empty([2,0,bottleneck.shape[2],bottleneck.shape[3]], device=bottleneck.device)
        skips.append(bottleneck)

        if debug_print_shapes:
            print("== Skip tensors provided to decoder ==")
            for i, s in enumerate(skips, start=1):
                print(f"skip {i}: shape {tuple(s.shape)}")
        dec_out = self.decoder(skips)
        # print(f"Decoder output shape before segmentation_head: {dec_out.shape}")

        out = self.segmentation_head(dec_out)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderUNet(
                            unet_encoder_weights="imagenet",
                            segformer_variant="nvidia/segformer-b0-finetuned-ade-512-512",
                            simple_fusion = 3,
                            ).to(device)

    print(model)  # print network structure

    x = torch.randn(1, 3, 1024, 1024).to(device)

    with torch.no_grad():
        out = model(x, debug_print_shapes=True)
        print("Output shape:", tuple(out.shape))
