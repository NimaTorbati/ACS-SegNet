import copy
import math

import numpy as np
import torch
import torch.nn as nn
from timm.layers import LayerNorm
from torch.nn import Conv2d, Dropout, Linear, Softmax

from src.network.transfomer_based.transUnet.vit_seg_modeling import ACT2FN

# U_Net+Transfomer+attention_skip
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)

        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class skip_attention(nn.Module):
    """
    Multi-scale attention gate
    """
    def __init__(self, channel):
        super(skip_attention, self).__init__()
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

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        avg_x1=torch.mean(x1,dim=1,keepdim=True)
        max_x2=torch.max(x2,dim=1,keepdim=True).values
        x3=self.Conv3(torch.cat((avg_x1,max_x2),dim=1))
        x4=self.sig(x3)
        return x+x*x4

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Transformer(nn.Module):
    def __init__(self, vis):
        super(Transformer, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        for _ in range(12):
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Encoder(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,vis=False):
        super(Encoder,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.patch_embeddings = Conv2d(in_channels=1024, #1024
                                       out_channels=768, #768
                                       kernel_size=1, #1
                                       stride=1) #1

        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 768))
        self.dropout = Dropout(0.1)
        self.transformer=Transformer(vis)


    def forward(self,x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        features=[x4,x3,x2,x1]
        x6=self.patch_embeddings(x5)
        x6 = x6.flatten(2) # (B*768*196)
        x6 = x6.transpose(-1, -2)  #  (B*196*768)
        x7 = x6 + self.position_embeddings
        x8 = self.dropout(x7)
        x9, attn_weights = self.transformer(x8)
        return x9,features

class Decoder(nn.Module):
    def __init__(self,output_ch=1):
        super(Decoder,self).__init__()
        self.Up5 = up_conv(ch_in=768, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.skip_attention5=skip_attention(channel=512)
        self.skip_attention4=skip_attention(channel=256)
        self.skip_attention3=skip_attention(channel=128)
        self.skip_attention2=skip_attention(channel=64)

    def forward(self,hidden_states,features):

        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        d5 = self.Up5(x)
        d5 = torch.cat((self.skip_attention5(features[0]), d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((self.skip_attention4(features[1]), d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((self.skip_attention3(features[2]), d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((self.skip_attention2(features[3]), d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


class one_encoder(nn.Module):
    def __init__(self, img_ch=3, output_ch=1,vis=False):
        super(one_encoder, self).__init__()

        self.encoder=Encoder(img_ch,output_ch,vis)
        self.decoder=Decoder(output_ch)


    def forward(self, x):


        x,features=self.encoder(x)
        x=self.decoder(x,features)

        # decode

        return x