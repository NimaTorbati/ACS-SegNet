import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, U, S):
        P = self.s_proj(S)
        M_cos = U * torch.cos(P)
        M_sin = U * torch.sin(P)
        fused = torch.cat([U, self.wa(M_cos) - self.wb(M_sin)], dim=1)

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
    def forward(self, U, S = None):
        # Resize S if needed to match U
        if S is None:
            x = U
        else:
            x = torch.cat([U, S], dim=1)

        fused = x  # Concatenate along channel dimension
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
