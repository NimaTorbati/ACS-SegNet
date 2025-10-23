import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

T = 5


# ==============================蒸馏损失===============================
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # student网络输出软化后结果
        # log_softmax与softmax没有本质的区别，只不过log_softmax会得到一个正值的loss结果。
        p_s = F.log_softmax(y_s, dim=1)

        # # teacher网络输出软化后结果
        p_t = F.softmax(y_t, dim=1)

        # 蒸馏损失采用的是KL散度损失函数
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss



class BCELoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(BCELoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = torch.sigmoid(y_s)
        p_t = torch.sigmoid(y_t)
        loss = F.binary_cross_entropy(p_s,p_t)
        return loss


class MSELoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_s, y_t):
        p_s = torch.sigmoid(y_s)
        p_t = torch.sigmoid(y_t)
        loss = F.mse_loss(p_s,p_t)
        return loss



class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        preds_T[0].detach()
        assert preds_S[0].shape == preds_T[0].shape, 'the output dim of teacher and student differ'
        N, C, W, H = preds_S[0].shape
        softmax_pred_T = F.softmax(preds_T[0].permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum(
            - softmax_pred_T * logsoftmax(preds_S[0].permute(0, 2, 3, 1).contiguous().view(-1, C)))) / W / H
        return loss


# # Loss functions
# class PerceptualLoss():
#     def contentFunc(self):
#         conv_3_3_layer = 14
#         cnn = models.vgg19(weights=True).features
#         cnn = cnn.cuda()
#         model = nn.Sequential()
#         model = model.cuda()
#         for i, layer in enumerate(list(cnn)):
#             model.add_module(str(i), layer)
#             if i == conv_3_3_layer:
#                 break
#         return model
#
#     def __init__(self, loss):
#         self.criterion = loss
#         self.contentFunc = self.contentFunc()
#
#     def get_loss(self, fakeIm, realIm):
#         f_fake = self.contentFunc.forward(fakeIm)
#         f_real = self.contentFunc.forward(realIm)
#         f_real_no_grad = f_real.detach()
#         loss = self.criterion(f_fake, f_real_no_grad)
#         return loss
