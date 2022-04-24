import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import trainer
from utils import ARGS

ResNet = models.resnet50

USE_SAVED_MODEL = True
SAVED_MODEL_PATH = 'checkpoints/checkpoint-pretrained_resnet_coco_focalloss_epochplus50-epoch50.pth'

# args = ARGS(epochs=15, batch_size=16, lr=0.001, inp_size=360, use_cuda=True, val_every=2000, save_at_end=True, save_freq=5, step_size=10, gamma=0.1)
args = ARGS(epochs=50, batch_size=8, lr=0.00001, inp_size=360, use_cuda=True, 
            val_every=2000, save_at_end=True, save_freq=1, step_size=40, gamma=0.1, 
            test_batch_size=10, indoor_only=True)
if args.indoor_only:
    num_classes = 14
else:
    num_classes = 90

if USE_SAVED_MODEL == False : 
    model = ResNet(pretrained=True)
    model.fc = nn.Linear(2048,num_classes)
else : 
    model = ResNet()
    model.fc = nn.Linear(2048,num_classes)
    print("Loading model....")
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    # if num_classes == 14:
        # model.fc = nn.Linear(2048,num_classes)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
test_ap, test_map, test_fraction_correct = trainer.train(args, model, optimizer, scheduler, 'pretrained_resnet_coco_focalloss_losssum_epochplus50_indoor')
trainer.test(args, model, 'test', log_wandb=True)