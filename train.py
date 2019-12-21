from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, accuracy, get_learning_rate, \
    adjust_learning_rate
import torch
import torch.nn as nn
import numpy as np
from net import CarRecognitionNet
from dataset import fetch_dataloaders
from config import device


def train(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = 0
    epochs_since_improvement = 0
    logger = get_logger()
    if checkpoint is None:
        model = CarRecognitionNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # 构建数据集
    loader = fetch_dataloaders(args.image_folder, [0.8, 0.2], batchsize=args.batch_size)
    train_loader = loader['train']
    valid_loader = loader['valid']

    for epoch in range(start_epoch, args.end_epoch):
        # 模型长时间不更新退出
        if epochs_since_improvement > 50:
            break

        # 调整学习率
        if epochs_since_improvement > 0 and epochs_since_improvement % 15 == 0:
            adjust_learning_rate(optimizer, 0.1)

        train_loss, train_acc = __train(train_loader=train_loader,
                                        model=model,
                                        optimizer=optimizer,
                                        epoch=epoch,
                                        logger=logger
                                        )

        vail_loss, vail_acc = __valid(valid_loader=valid_loader,
                                      model=model,
                                      logger=logger)
        is_best = vail_acc > best_acc
        best_acc = max(vail_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            logger.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            # Save checkpoint
        save_checkpoint("OURS_RES50", epoch, epochs_since_improvement, model, optimizer, best_acc, is_best)


def __train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode

    losses = AverageMeter('Loss', ':.5f')
    accs = AverageMeter('Acc', ':.5f')

    for index, items in enumerate(train_loader):
        # images[0] 是一个batch的图片
        # images[1] 是一个batch对应的label
        images = items[0].to(device)
        label_model = items[1].to(device) # 一类 二类
        label_type = (items[1] // 4).clamp_(min=0, max=1).to(device)  # 货车 或者 客车
        # print(label_type)

        # 计算loss
        output_type, output_model = model(images)
        # print(output_type)
        # print(output_model)
        loss_type = nn.NLLLoss()(output_type, label_type)
        loss_model = nn.NLLLoss()(output_model, label_model)
        total_loss = 0.5*loss_type + loss_model

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()

        # 更新权值
        optimizer.step()

        # 计算acc
        # 只计算最后车型的准确度，以评估模型的能力
        pred_model = output_model.data.max(1)[1] # 返回值是索引 tensor([7, 5, 3])
        # 计算model的准确度
        acc = accuracy(pred_model, label_model)

        # 追踪更新
        losses.update(total_loss.item())
        accs.update(acc)

        if index % 10 == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                     'Accuracy {acc.val:.5f} ({acc.avg:.5f})\t'.format(epoch, index,
                                                                       len(train_loader),
                                                                       loss=losses,
                                                                       acc=accs,
                                                                       )
            logger.info(status)

    return losses.avg, accs.avg


def __valid(valid_loader, model, logger):
    model.eval()  # eval mode

    losses = AverageMeter('Loss', ':.5f')
    accs = AverageMeter('Acc', ':.5f')

    for index, items in enumerate(valid_loader):
        # images[0] 是一个batch的图片
        # images[1] 是一个batch对应的label
        images = items[0].to(device)

        label_type = (items[1] // 4).clamp_(min=0, max=1).to(device)  # 0 1
        label_model = items[1].to(device)  # 0 ~ 9

        with torch.no_grad():
            output_type, output_model = model(images)

        # 计算loss
        loss_type = nn.NLLLoss()(output_type, label_type)
        loss_model = nn.NLLLoss()(output_model, label_model)
        total_loss = 0.5*loss_type + loss_model

        # 计算model的准确度
        pred_model = output_model.data.max(1)[1]  # 返回值是索引 tensor([7, 5, 3])、
        acc = accuracy(pred_model, label_model)

        losses.update(total_loss.item())
        accs.update(acc)

        # Print status
    status = 'Validation\t Loss {loss.avg:.5f}\t Accuracy {acc.avg:.5f}\n'.format(loss=losses, acc=accs)
    logger.info(status)

    return losses.avg, accs.avg


def main():
    global args
    print("======训练开始======")
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
