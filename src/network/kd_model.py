from torch import nn

import torch
from src.network.conv_based.AMSUnet import AMSUnet
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.CGNet import Context_Guided_Network
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.ELUnet import ELUnet
from src.network.conv_based.ESPNet import ESPNet
from src.network.conv_based.ULite import ULite
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.cfpnet import CFPNet
from src.network.conv_based.ddrnet import DDRNet
from src.network.conv_based.pp_liteseg import PPLiteSeg
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model


def get_model(name, out_ch):
    if name == "CMUNet":
        model = CMUNet(output_ch=out_ch).cuda()
    elif name == "CMUNeXt":
        model = cmunext(num_classes=out_ch).cuda()
    elif name == "U_Net":
        model = U_Net(output_ch=out_ch).cuda()
    elif name == "AttU_Net":
        model = AttU_Net(output_ch=out_ch).cuda()
    elif name == "UNext":
        model = UNext(output_ch=out_ch).cuda()
    elif name == ("UNetplus"):
        model = ResNet34UnetPlus(num_class=out_ch).cuda()
    elif name == "UNet3plus":
        model = UNet3plus(n_classes=out_ch).cuda()
    elif name == "ESPNetv2":
        model = ESPNet().cuda()
    elif name == "ELUnet":
        model = ELUnet().cuda()
    elif name == "AMSUnet":
        model = AMSUnet().cuda()
    elif name == "ULite":
        model = ULite().cuda()
    elif name == "CGNet":
        model = Context_Guided_Network().cuda()
    elif name == "PPLiteSeg":
        model = PPLiteSeg().cuda()
    elif name == "CFPNet":
        model = CFPNet().cuda()
    elif name == "DDRNet":
        model = DDRNet().cuda()
    else:
        model = get_transformer_based_model(parser=None, model_name=name, img_size=256,
                                            num_classes=1, in_ch=3).cuda()

    return model


class kd_model(nn.Module):
    def __init__(self, output_ch=1, teacher_model=None, student_model=None):
        super(kd_model, self).__init__()
        self.teacher_model = get_model(teacher_model, output_ch)
        self.teacher_model.load_state_dict(torch.load('checkpoint/{}_model.pth'.format(teacher_model)))
        self.student_model = get_model(student_model, output_ch)

    def forward(self, x):
        teacher_out = self.teacher_model(x).detach()
        student_out = self.student_model(x)
        return teacher_out, student_out

    def get_parameters(self, x):
        if x == "teacher":
            return self.teacher_model.parameters()
        else:
            return self.student_model.parameters()
