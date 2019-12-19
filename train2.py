import numpy as np
import torch
from torch import nn

from dataset import fetch_dataloaders

from config import device
from net import BaseModel
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, accuracy, get_learning_rate, \
    adjust_learning_rate


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = 0
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = BaseModel()
        # model = nn.DataParallel(model)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Custom dataloaders
    # 构建数据集
    loader = fetch_dataloaders(args.image_folder, [0.8, 0.2], batchsize=args.batch_size)
    train_loader = loader['train']
    valid_loader = loader['valid']
    # scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        if epochs_since_improvement > 50:
            break
        # Decay learning rate if there is no improvement for 10 consecutive epochs
        if epochs_since_improvement > 0 and epochs_since_improvement % 12 == 0:
            adjust_learning_rate(optimizer, 0.1)

        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)

        lr = get_learning_rate(optimizer)

        # One epoch's validation
        valid_loss, valid_acc = valid(valid_loader=valid_loader,
                                      model=model,
                                      criterion=criterion,
                                      logger=logger)

        # Check if there was an improvement
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            logger.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint("RES152", epoch, epochs_since_improvement, model, optimizer, best_acc, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter('Loss', ':.5f')
    accs = AverageMeter('Acc', ':.5f')

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)  # [N, 3, 224, 224]
        label = label.to(device)  # [N, 196]

        # Forward prop.

        out = model(img)

        # Calculate loss
        pred_type = out.data.max(1)[1]
        loss = criterion(out, label)
        acc = accuracy(pred_type, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

        # Print status
        if i % 10 == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                     'Accuracy {acc.val:.5f} ({acc.avg:.5f})\t'.format(epoch, i,
                                                                       len(train_loader),
                                                                       loss=losses,
                                                                       acc=accs,
                                                                       )
            logger.info(status)

    return losses.avg, accs.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter('Loss', ':.5f')
    accs = AverageMeter('Acc', ':.5f')

    # Batches
    for i, (img, label) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.to(device)  # [N, 3, 224, 224]
        label = label.to(device)  # [N, 196]

        # Forward prop.
        with torch.no_grad():
            out = model(img)

        # Calculate loss
        loss = criterion(out, label)
        pred_type = out.data.max(1)[1]
        acc = accuracy(pred_type, label)

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

    # Print status
    status = 'Validation\t Loss {loss.avg:.5f}\t Accuracy {acc.avg:.5f}\n'.format(loss=losses, acc=accs)
    logger.info(status)

    return losses.avg, accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
