# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import models
import logging
import time
from utils.metrics import AverageMeter, ProgressMeter, accuracy
import random
import mlflow

class LinearSigmoid(torch.autograd.Function):
    """
    Surrogate gradient based on arctan, used in Feng et al. (2021)
    """
    @staticmethod
    def forward(ctx, x):
        result = torch.zeros_like(x)
        # Segment 1: x <= -2 -> approximate with 0
        result = torch.where(x <= -2, torch.zeros_like(x), result)
        # Segment 2: -2 < x < 2 -> approximate with 0.25 * x + 0.5
        result = torch.where((x > -2) & (x < 2), 0.25 * x + 0.5, result)
        # Segment 3: x >= 2 -> approximate with 1
        result = torch.where(x >= 2, torch.ones_like(x), result)
        return result

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None

def custom_mse_loss(x, labels):
    x = LinearSigmoid.apply(x)
    loss = F.mse_loss(x, F.one_hot(labels, 11).float())
    return loss

def train(args, device, train_loader, test_loader):
    if args.seed != 0:
        set_seed(args.seed)

    for trial in range(1, args.trials + 1):

        # Network topology
        model = models.__dict__[args.arch](args, device)
        if trial == 1:
            logging.info(f'Total Parameters: {int(10*(sum(p.numel() for p in model.parameters()) / 1000.0))/10}K')
        # Use CUDA for GPU-based computation if enabled
        if args.cuda:
            model.cuda()

        # Initial monitoring
        if (args.trials > 1):
            logging.info('\nIn trial {} of {}'.format(trial, args.trials))
        if (trial == 1):
            logging.info("=== Model ===")
            logging.info(model)

        # Optimizer
        if "bptt" == args.training_mode:
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'NAG':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
            elif args.optimizer == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
            elif args.optimizer == 'RProp':
                optimizer = optim.Rprop(model.parameters(), lr=args.lr)
            else:
                raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        else:
            # This optimizer is only for the local learning methods such as LLS and LocalLosses
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(model.linear.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'Adam':
                optimizer = optim.Adam(model.linear.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.scheduler > 0:
            if args.lr_scheduler_type == "Cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler)
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        else:
            scheduler = None

        if args.loss == 'MSE':
            loss = nn.MSELoss()
        elif args.loss == 'BCE':
            loss = nn.BCELoss()
        elif args.loss == 'CE':
            loss = nn.CrossEntropyLoss()
        elif args.loss == 'MSEHW':
            loss = custom_mse_loss
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        # Training and performance monitoring
        logging.info("\n=== Starting model training with %d epochs:\n" % (args.epochs,))
        best_acc1 = 0
        loss_val = 0
        acc_train_hist = []
        acc_val_hist = []
        loss_train_hist = []
        loss_val_hist = []
        for epoch in range(1, args.epochs + 1):
            logging.info("\t Epoch " + str(epoch) + "...")
            try:
                lr = scheduler.get_last_lr()[0]
            except:
                lr = args.lr
            # Will display the average accuracy on the training set during the epoch (changing weights)
            acc_t, loss_t = do_epoch(args, True, model, device, train_loader, optimizer, loss, 'train', epoch, lr)
            acc_train_hist.append(acc_t.cpu().numpy())
            loss_train_hist.append(loss_t)
            mlflow.log_metric("acc_train", acc_t, step=epoch)
            mlflow.log_metric("loss_train", loss_t, step=epoch)

            # Check performance on the training set and on the test set:
            if not args.skip_test:
                acc1, loss_val = do_epoch(args, False, model, device, test_loader, optimizer, loss, 'test', epoch)
                acc_val_hist.append(acc1.cpu().numpy())
                loss_val_hist.append(loss_val)
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
                mlflow.log_metric("acc_val", acc1, step=epoch)
                mlflow.log_metric("loss_val", loss_val, step=epoch)
                mlflow.log_metric("best_acc", best_acc1)
                if is_best:
                    if is_best:
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(state, args.save_path + f'/trial_{trial}_model_best.pth.tar')
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss_val)
                else:
                    scheduler.step()
                if hasattr(model, 'scheduler_step'):
                    model.scheduler_step(loss_val)
                logging.info(f'Last learning rate: {scheduler.get_last_lr()}')
                mlflow.log_metric("lr_value", scheduler.get_last_lr()[0], step=epoch)


def do_epoch(args, do_training: bool, model, device, loader, optimizer, loss_fct, benchType, epoch, lr=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if benchType == 'train':
        progress = ProgressMeter(
            len(loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

    if not do_training:
        model.eval()
    else:
        model.train()
    score = 0
    loss = 0
    batch = args.batch_size if (benchType == 'train') else args.val_batch_size
    length = args.full_train_len if (benchType == 'train') else args.full_test_len

    end = time.time()
    for batch_idx, (data, label) in enumerate(loader):
        data_time.update(time.time() - end)

        label = label.type(torch.int64)
        data, label = data.float().to(device), label.to(device)

        data, label, target, timesteps = data_resizing(args, data, label, device)

        args.n_steps = timesteps

        model.reset_states()
        if not do_training:
            with torch.no_grad():
                pred = 0
                for t in range(args.n_steps):
                    input = data[t] if data.size(0) > 1 else data[0]
                    output = model(input, None)
                    pred += output
        elif args.training_mode == "bptt":
            optimizer.zero_grad()
            pred = 0
            for t in range(args.n_steps):
                input = data[t] if data.size(0) > 1 else data[0]
                pred += model(input, target)
            loss = loss_fct(pred, label)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            if hasattr(model, "optimizer_zero_grad"):
                model.optimizer_zero_grad()
            pred = 0
            for t in range(args.n_steps):
                input = data[t] if data.size(0) > 1 else data[0]
                output = model(input, target=None if (data.size(0) - t) <= args.delay_ls else target)
                pred += output.detach()
                if (data.size(0) - t) <= args.delay_ls:
                    loss = loss_fct(output, label)
                    loss.backward()
            optimizer.step()
            if hasattr(model, "optimizer_step"):
                model.optimizer_step()

        with torch.no_grad():
            loss = loss_fct(pred, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(pred, label, topk=(1, 5))
        losses.update(loss.item(), data.size(1))
        top1.update(acc1[0], data.size(1))
        top5.update(acc5[0], data.size(1))

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_freq == (args.print_freq-1):
            progress.display(batch_idx)

    if benchType == 'train':
        logging.info(' @Training * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(top1=top1, top5=top5, loss=losses))
    else:
        logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(top1=top1, top5=top5, loss=losses))
    return top1.avg, losses.avg


def data_resizing(args, data, label, device):
    timesteps = data.size(1)
    batch_size = data.size(0)
    if args.dataset == 'DVSGesture':
        data = data.view(batch_size, timesteps, 2, 32, 32)
        data = data.permute(1, 0, 2, 3, 4)
        if timesteps > 22:
            timesteps = 20
        # label = label.unsqueeze(1).expand(batch_size, timesteps)
    elif args.dataset == 'CIFAR10DVS':
        data = data.view(batch_size, timesteps, 2, 48, 48)
        data = data.permute(1, 0, 2, 3, 4)
        # label = label.unsqueeze(1).expand(batch_size, timesteps)
    elif args.dataset == 'CIFAR10':
        data = data.view(batch_size, 1, 3, 32, 32)
        data = data.permute(1, 0, 2, 3, 4)
        timesteps = args.n_steps
    elif args.dataset == 'CIFAR100':
        data = data.view(batch_size, 1, 3, 32, 32)
        data = data.permute(1, 0, 2, 3, 4)
        timesteps = args.n_steps
    else:
        logging.info("ERROR: {0} is not supported".format(args.dataset))
        raise NameError("{0} is not supported".format(args.dataset))

    if args.classif and args.label_encoding == "one-hot":  # Do a one-hot encoding for classification
        target = F.one_hot(label, num_classes=args.n_classes).float()

    else:
        target = label
    label = label.view(-1, )

    return data, label, target, timesteps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True