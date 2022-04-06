import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import trainer
from utils import ARGS

num_classes = 90
ResNet = models.resnet50

args = ARGS(epochs=15, batch_size=16, lr=0.001, inp_size=224, use_cuda=True, val_every=250, save_at_end=True, save_freq=9, step_size=21, gamma=0.1)

model = ResNet(pretrained=True)
model.fc = nn.Linear(2048,num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
test_ap, test_map = trainer.train(args, model, optimizer, scheduler, 'pretrained_resnet_coco')