import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from config import device

word_embedding_dim = 2048
hidden_dim = 2048
num_layers = 1
num_model_class = 2
num_type_class = 10


def remove_last(model):
    new_classifer = nn.Sequential(*list(model.children())[:-2])
    return new_classifer


class CarRecognitionNet(nn.Module):
    def __init__(self):
        super(CarRecognitionNet, self).__init__()
        # 加载在imagenet上预训练好的模型
        model = models.resnet50(pretrained=True)
        # model.load_state_dict(torch.load("./resnet_pth/resnet152.pth"))
        # 去除最后2层
        self.resnet = remove_last(model)

        # 定义CarRecognitionNet的操作
        self.first_gap = nn.AvgPool2d(kernel_size=(7, 7), stride=(7, 7))
        self.second_gap = nn.AvgPool2d(kernel_size=(7, 7), stride=(7, 7))

        self.relu = nn.ReLU(inplace=True)
        self.attention_activation = nn.Softplus()

        self.epsilon = 1e-1
        self.rnn = nn.GRU(word_embedding_dim, hidden_dim, num_layers, bidirectional=False)

        self.W1 = nn.Linear(hidden_dim, num_model_class)
        init.kaiming_normal_(self.W1.weight)

        self.W2 = nn.Linear(hidden_dim, num_type_class)
        init.kaiming_normal_(self.W2.weight)

        self.W3 = nn.Linear(hidden_dim, word_embedding_dim)
        init.kaiming_normal_(self.W3.weight)

        self.W4 = nn.Linear(hidden_dim, word_embedding_dim)
        init.kaiming_normal_(self.W4.weight)

        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, im):
        h0 = torch.zeros((num_layers, len(im), hidden_dim), device=device)
        im_feat = self.resnet(im)
        # print()
        first_im_feat = self.first_gap(im_feat)
        first_im_feat = first_im_feat.view(-1, word_embedding_dim)
        first_im_feat = torch.unsqueeze(first_im_feat, 0)
        output, hn = self.rnn(first_im_feat, h0)
        # 车型
        o_type = output[0, :, :]

        o_type_input = self.W3(o_type)
        o_type_input = self.relu(o_type_input)
        o_type_input = self.W4(o_type_input)
        o_type_input_expand = o_type_input.unsqueeze(2).unsqueeze(2).expand_as(im_feat)

        second_im_score = torch.mul(im_feat, o_type_input_expand)
        second_im_score = torch.sum(second_im_score, dim=1, keepdim=True)
        second_im_score = self.attention_activation(second_im_score)  # relu的平滑版本

        second_im_score = second_im_score + self.epsilon
        second_im_score_total = torch.sum(second_im_score.view(-1, 49), dim=1, keepdim=True)
        second_im_score_normalized = torch.div(second_im_score,
                                               second_im_score_total.unsqueeze(2).unsqueeze(2)
                                               .expand_as(second_im_score))

        second_im_feat = torch.mul(im_feat, second_im_score_normalized.expand_as(im_feat))

        # 上面都是attention
        second_im_feat = self.second_gap(second_im_feat)
        second_im_feat = second_im_feat.view(-1, word_embedding_dim).unsqueeze(0)
        output, hn = self.rnn(second_im_feat, hn)

        # 种类
        o_model = output[0, :, :]

        pred_type = self.W1(o_type)
        pred_model = self.W2(o_model)
        return self.LogSoftmax(pred_type), self.LogSoftmax(pred_model),


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        resnet = models.resnet152(pretrained=False)
        resnet.load_state_dict(torch.load("./resnet_pth/resnet152.pth"))
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        x = self.softmax(x)
        return x
