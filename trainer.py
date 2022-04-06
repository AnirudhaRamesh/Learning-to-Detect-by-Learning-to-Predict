from __future__ import print_function

import torch
import numpy as np

import utils
# from voc_dataset import VOCDataset
from coco import CocoDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
focal_loss = FocalLoss()


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    # TODO: Q2.2 Implement code for model saving
    filename = 'checkpoints/' +'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    torch.save(model.state_dict(), filename)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO Q1.5: Initialize your tensorboard writer here!
    
    writer = SummaryWriter()
    
    # train_loader = utils.get_data_loader(
    #     'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    # test_loader = utils.get_data_loader(
    #     'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    train_dataset = CocoDataset('/mnt/aidtr/external/coco/train2017', '/mnt/aidtr/external/coco/annotations/instances_train2017.json', args.inp_size)
    val_dataset = CocoDataset('/mnt/aidtr/external/coco/val2017', '/mnt/aidtr/external/coco/annotations/instances_val2017.json', args.inp_size)

    bce_loss = torch.nn.BCEWithLogitsLoss()

    # own DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,)
                                            # collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=4,)
                                            # collate_fn=collate_fn)
    # Ensure model is in correct mode and on right device
    model.train()
    print(args.device)
    model = model.to(args.device)

    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get a batch of data
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            print(data.shape)
            output = model(data)
            # Calculate the loss
            # TODO Q1.4: your loss for multi-label classification
#             loss = F.cross_entropy(output, target)
#             print(output.shape, target.shape)
            # loss = focal_loss(output, target)
            loss = bce_loss(output, target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # TODO Q1.5: Log training loss to tensorboard
                writer.add_scalar('Loss/train', loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                # TODO Q3.2: Log histogram of gradients
                for name, param in model.named_parameters():
#                     if 'bn' not in name:            
                    writer.add_histogram('Weights/'+name, param, cnt)
                    writer.add_histogram('Gradients/'+name, param.grad, cnt)
                        
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print('Val Epoch: {} [{} ({:.0f}%)]\map: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), map))
                # TODO Q1.5: Log MAP to tensorboard
                writer.add_scalar('MAP/test', map, cnt)
                model.train()
            cnt += 1

        # TODO Q3.2: Log Learning rate
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('LearningRate/lr', scheduler.state_dict()['_last_lr'][0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map