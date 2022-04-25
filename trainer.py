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

import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
import os

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        # ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        # ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction, weight=self.weight)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='sum', weight=self.weight)
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
    # print(f"/mnt/aidtr/external/coco/annotations/instances_val2017{'indoor' if args.indoor_only else ''}.json")

    try:
        train_dataset = CocoDataset('/mnt/aidtr/external/coco/train2017', f"/mnt/aidtr/external/coco/annotations/instances_train2017{'indoor' if args.indoor_only else ''}.json", args.inp_size, args.indoor_only)
        val_dataset = CocoDataset('/mnt/aidtr/external/coco/val2017', f"/mnt/aidtr/external/coco/annotations/instances_val2017{'indoor' if args.indoor_only else ''}.json", args.inp_size, args.indoor_only)
    except:
        print("You don't have the indoor annotated dataset ready, preparing it for you! ... . . . .. ")
        os.system("python filter.py --input_json /mnt/aidtr/external/coco/annotations/instances_val2017.json --output_json /mnt/aidtr/external/coco/annotations/instances_val2017indoor.json --categories book clock vase scissors 'teddy bear' 'hair drier' toothbrush bottle 'wine glass' cup fork knife spoon bowl")
        os.system("python filter.py --input_json /mnt/aidtr/external/coco/annotations/instances_train2017.json --output_json /mnt/aidtr/external/coco/annotations/instances_train2017indoor.json --categories book clock vase scissors 'teddy bear' 'hair drier' toothbrush bottle 'wine glass' cup fork knife spoon bowl")
        train_dataset = CocoDataset('/mnt/aidtr/external/coco/train2017', f"/mnt/aidtr/external/coco/annotations/instances_train2017{'indoor' if args.indoor_only else ''}.json", args.inp_size, args.indoor_only)
        val_dataset = CocoDataset('/mnt/aidtr/external/coco/val2017', f"/mnt/aidtr/external/coco/annotations/instances_val2017{'indoor' if args.indoor_only else ''}.json", args.inp_size, args.indoor_only)
    
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
    best = -1
    for epoch in range(args.epochs):
        for batch_idx, (data, target, _) in tqdm(enumerate(train_loader)):
            # Get a batch of data
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO Q1.4: your loss for multi-label classification
#             loss = F.cross_entropy(output, target)
#             print(output.shape, target.shape)
            loss = focal_loss(output, target)
            # loss = bce_loss(output, target)
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
                # for name, param in model.named_parameters():
#                     if 'bn' not in name:            
                    # writer.add_histogram('Weights/'+name, param, cnt)
                    # writer.add_histogram('Gradients/'+name, param.grad, cnt)
                        
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map, fraction_correct = utils.eval_dataset_map(model, args.device, test_loader)
                print('Val Epoch: {} [{} ({:.0f}%)]\map: {:.6f} | acc: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), map, fraction_correct))
                # TODO Q1.5: Log MAP to tensorboard
                if best < map:
                    save_model(0, "best_" + model_name, model)
                    best = map
                    print("Saving best model...")
                writer.add_scalar('MAP/test', map, cnt)
                # writer.add_scalar('AP/test', torch.Tensor(ap), cnt)
                writer.add_scalar('PercentageCorrect/test', fraction_correct * 100, cnt)
                model.train()
            cnt += 1

        # TODO Q3.2: Log Learning rate
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('LearningRate/lr', scheduler.state_dict()['_last_lr'][0], cnt)

        # save model
        # if save_this_epoch(args, epoch):
            # save_model(epoch, model_name, model)

    # Validation iteration
    # test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    ap, map, fraction_correct = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map, fraction_correct

def test(args, model, model_name='model', log_wandb=False):

    val_dataset = CocoDataset('/mnt/aidtr/external/coco/val2017', f"/mnt/aidtr/external/coco/annotations/instances_val2017{'indoor' if args.indoor_only else ''}.json", args.inp_size, args.indoor_only)
    test_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            num_workers=4,)
    print("length test loader : ", len(test_loader))
    model.eval()
    print(args.device)
    model = model.to(args.device)

    if log_wandb :
        wandb.init(project="vlr-project-trained-class-predictor", reinit=True)

    with torch.no_grad() : 
        for batch_idx, (data, target, display_img) in enumerate(test_loader):
            # Get a batch of data
            data, target = data.to(args.device), target.to(args.device)
            # Forward pass
            output = model(data)

            output = torch.argmax(output, dim=1)
            target = torch.argmax(target, dim=1)

            labels = val_dataset.class_to_label(output.cpu().numpy())
            labels_gt = val_dataset.class_to_label(target.cpu().numpy())

            if log_wandb :
                caption_full = 'predicted class : ' + labels[0] + '\n gt class : ' + labels_gt[0]
                img = wandb.Image(display_img[0], caption=caption_full)
                # wandb.log({'image_{}'.format(batch_idx): img})
                wandb.log({'image' : img})