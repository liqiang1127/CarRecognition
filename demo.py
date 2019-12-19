import numpy as np
import torch
import torch.nn as nn
from utils import accuracy
import torchvision.models as models
from torchscope import scope
from net import CarRecognitionNet
from config import device
from utils import AverageMeter

losses = AverageMeter('Loss', ':.5f')
accs = AverageMeter('Acc', ':.5f')

for i in range(10):
    losses.update(1 * i)
    accs.update(2 * i)
    print('Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                     'Accuracy {acc.val:.5f} ({acc.avg:.5f})\t'.format(i, i,
                                                                       100,
                                                                       loss=losses,
                                                                       acc=accs,
                                                                       ))
# x = np.random.rand(3, 3)
# # print(x)
# y = torch.tensor(x)
# print(y)
# pred_type = y.data.max(1)[1]
# print(pred_type)

# print("===========================")
# target = torch.tensor([1,2,4])
# print(pred_type)
# acc = accuracy(pred_type, target)
# print(acc)

# model = models.resnet152(pretrained=False)
# car = CarRecognitionNet().to(device)
# scope(car, input_size=(3, 224, 224), device='cuda')
# # print(list(model.children())[-3])
# print(list(model.children())[-2])
# print(list(model.children())[-1])

x = torch.tensor([1, 1])
y = torch.tensor([[-0.7564, -0.6337],
                  [-0.6932, -0.6931]])
nn.NLLLoss()(y, x)
