from torch import nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18, alexnet, densenet121
from torchvision.models import mobilenet_v3_small, efficientnet_v2_s
from torchinfo import summary
#nn.Moduleを見る

def get_densenet(args):
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(in_features=1024, out_features=10)
    if args.dataset == "MNIST" or args.dataset == "FMNIST":
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)

    return model

def get_mobilenet(args):
    model = mobilenet_v3_small(pretrained=True)
    if args.dataset == "MNIST" or args.dataset == "FMNIST":
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[3] = nn.Linear(in_features=1024, out_features=10)
    # print(model.features)
    # exit(0)
    # model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
    return model

def get_effientnet():
    model = efficientnet_v2_s()
    model.classifier[1] = nn.Linear(in_features=1280, out_features=10)
    return model

def get_ghostnet():
    model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    print(model.classifier)
    model.classifier = nn.Linear(in_features=1280, out_features=10)
    
    return model

def get_alexnet():
    model = alexnet(pretrained=True)
    return model
def get_resnet(pretrained, args) -> nn.Module:
   # ImageNetで事前学習済みの重みをロード
   model = resnet18(pretrained=pretrained)

   # ここで更新する部分の重みは初期化される
   if args.dataset == "MNIST" or args.dataset == "FMNIST":
        model.conv1 = nn.Conv2d(
       in_channels=1,
       out_channels=64,
       kernel_size=model.conv1.kernel_size,
       stride=model.conv1.stride,
       padding=model.conv1.padding,
       bias=False
    )

   model.fc = nn.Linear(
       in_features=model.fc.in_features,
       out_features=10
   )
   return model

