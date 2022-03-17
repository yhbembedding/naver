import argparse

from lib import *
from loss import *
from dataset import *
from chart import *
from train_function import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["unet-resnet", "fcn","deeplabv3"], default="none")
parser.add_argument("--loss", choices=["focal"],default="none")
args = parser.parse_args()
if args.model == "unet-resnet":
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=24, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
elif args.model == "deeplabv3":
    model = smp.DeepLabV3(encoder_name='resnet34', encoder_depth=5,
                          encoder_weights='imagenet', decoder_channels=256, in_channels=3, classes=24,
                          activation='sigmoid', upsampling=8, aux_params=None)
elif args.model =="none":
    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=24, activation=None, encoder_depth=5,
                     decoder_channels=[256, 128, 64, 32, 16])

max_lr = 1e-3
epoch = 15
weight_decay = 1e-4
if args.loss =="focal":
    criterion = FocalLoss(gamma=0.1)
else:
    criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

history = train(epoch, model, train_loader, val_loader, criterion, optimizer, sched)

plot_loss(history)
plot_score(history)
plot_acc(history)

