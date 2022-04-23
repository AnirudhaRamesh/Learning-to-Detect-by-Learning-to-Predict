import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import trainer
from utils import ARGS

num_classes = 90
ResNet = models.resnet50

USE_SAVED_MODEL = True
SAVED_MODEL_PATH = 'checkpoints/checkpoint-pretrained_resnet_coco_focalloss_losssum-epoch35.pth'

# args = ARGS(epochs=15, batch_size=16, lr=0.001, inp_size=360, use_cuda=True, val_every=2000, save_at_end=True, save_freq=5, step_size=10, gamma=0.1)
args = ARGS(epochs=50, batch_size=16, lr=0.00001, inp_size=360, use_cuda=True, val_every=2000, save_at_end=True, save_freq=5, step_size=40, gamma=0.1, test_batch_size=10)


if USE_SAVED_MODEL == False : 
    model = ResNet(pretrained=True)
    model.fc = nn.Linear(2048,num_classes)
else : 
    model = ResNet()
    model.fc = nn.Linear(2048,num_classes)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
# test_ap, test_map, test_fraction_correct = trainer.train(args, model, optimizer, scheduler, 'pretrained_resnet_coco_focalloss_losssum_epochplus50')
trainer.test(args, model, 'test', log_wandb=True)