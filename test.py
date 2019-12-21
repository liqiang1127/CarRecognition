from utils import gen_csv
import torch
from PIL import Image
import argparse
import os
import numpy as np
from config import device
from dataset import data_transforms
from net import CarRecognitionNet
# --src=D:\\test

# 通过label2索引到结果的车型值
dic = {'0': 11, '1': 12, '2': 13, '3': 14, '4': 21, '5': 22, '6': 23, '7': 24, '8': 25, '9': 26}


def main(model):
    # 读取图片
    ret = []
    model.eval()
    for index in range(6):
        filename = "pic" + str(index+1) + '.jpg'
        file_path = os.path.join("./test", filename)
        image = Image.open(file_path)
        image = data_transforms['valid'](image)
        image = image.to(device)
        image.unsqueeze_(dim=0)
        # print(image)
        with torch.no_grad():
            _, pred_model = model(image)
        preds = pred_model.cpu().numpy()[0]
        prob = np.max(preds)
        class_id = np.argmax(preds)
        print("图片：{}\t预测结果：{}\t, log概率：{}".format(filename, class_id, prob))
        ret.append(class_id)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint)
    model = checkpoint['model']
    # optimizer = checkpoint['optimizer']
    print("以加载模型, acc = {}".format(checkpoint['acc']))
    ret = main(model)
    result = []
    for i in ret:
        result.append(dic[str(i)])
    gen_csv(result)
