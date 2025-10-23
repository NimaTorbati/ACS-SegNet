import functools

import numpy as np
import torch

from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torch import nn

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class TransUnet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(TransUnet, self).__init__()
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = output_ch
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))
        self.net = ViT_seg(config_vit, img_size=256, num_classes=output_ch).cuda()
        self.criterion=GANLoss().cuda()
        self.D = Discriminator()

    def forward(self, x,x_and_label = None):
        logits, pre_logits, features1, features2=self.net(x,x_and_label)
        pre_features1,pre_features2=self.D(features1.detach()),self.D(features2.detach())
        D_loss = self.criterion(pre_features1, False) * 0.5 + self.criterion(pre_features2, True) * 0.5
        encoder_loss = self.criterion(self.D(features1), True)
        return logits, pre_logits,D_loss,encoder_loss
        # return logits, pre_logits, features1, features2

    def val(self,x):
        return self.net.val(x)

    def get_parameters(self,net):
        if net=="seg_net":
            return self.net.parameters()
        elif net=="encoder":
            return self.net.transformer.parameters()
        else:
            return self.D.parameters()


class Discriminator(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence,)

    def forward(self, hidden_states):

        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        """Standard forward."""
        x=self.model(x)
        return x