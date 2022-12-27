from torch import nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18, alexnet, densenet121
from torchvision.models import mobilenet_v3_small, efficientnet_v2_s
from torchinfo import summary
#nn.Moduleを見る

def get_densenet():
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(in_features=1024, out_features=10)
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


class SE_Block(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_channels, int(in_channels*r))
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(int(in_channels*r), in_channels)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], out.shape[1], 1,1).expand_as(x)
        return out * x

class mobilenetv3_block(nn.Module):
    def __init__(self, in_channels, exp_size, out_channels, SE, NL, s, r=0.25):
        super().__init__()        
        if NL == 'HS':
            activation = nn.Hardswish
        else:
            activation = nn.ReLU

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=exp_size, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.relu1 = activation(False)

        self.conv2 = nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size = 3, stride=s, groups=exp_size, padding = 1)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.relu2 = activation(False)

        self.conv3 = nn.Conv2d(in_channels=exp_size, out_channels=out_channels, kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = activation(False)

        if SE:
            self.se = SE_Block(in_channels=out_channels, r=r)
        else:
            self.se = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        if self.se:
            out = self.se(out)

        if out.shape == x.shape:
            out = out + x

        return out
    
class MobileNetV3_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.Hardswish()

        self.bneck1  = mobilenetv3_block(in_channels=16, exp_size=16,  out_channels=16, SE=True,  NL='RE', s=2)
        self.bneck2  = mobilenetv3_block(in_channels=16, exp_size=72,  out_channels=24, SE=False, NL='RE', s=2)
        self.bneck3  = mobilenetv3_block(in_channels=24, exp_size=88,  out_channels=24, SE=False, NL='RE', s=1)
        self.bneck4  = mobilenetv3_block(in_channels=24, exp_size=96,  out_channels=40, SE=True,  NL='HS', s=2)
        self.bneck5  = mobilenetv3_block(in_channels=40, exp_size=240, out_channels=40, SE=True,  NL='HS', s=1)
        self.bneck6  = mobilenetv3_block(in_channels=40, exp_size=240, out_channels=40, SE=True,  NL='HS', s=1)
        self.bneck7  = mobilenetv3_block(in_channels=40, exp_size=120, out_channels=48, SE=True,  NL='HS', s=1)
        self.bneck8  = mobilenetv3_block(in_channels=48, exp_size=144, out_channels=48, SE=True,  NL='HS', s=1)
        self.bneck9  = mobilenetv3_block(in_channels=48, exp_size=288, out_channels=96, SE=True,  NL='HS', s=2)
        self.bneck10 = mobilenetv3_block(in_channels=96, exp_size=576, out_channels=96, SE=True,  NL='HS', s=1)
        self.bneck11 = mobilenetv3_block(in_channels=96, exp_size=576, out_channels=96, SE=True,  NL='HS', s=1)

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(576)
        self.relu2 = nn.Hardswish()

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(576, 1024)
        self.relu3 = nn.Hardswish()
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.bneck1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.bneck7(x)
        x = self.bneck8(x)
        x = self.bneck9(x)
        x = self.bneck10(x)
        x = self.bneck11(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x



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

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        '''
        input:
            dim_input: dimension of input
            dim_hidden: dimension of hidden layer
            dim_output: dimension of output
        '''
        super().__init__()
        self.layer_input = nn.Linear(dim_input, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


# class CNNMnist(nn.Module):
#     '''
#     input:
#         num_channels: number of channel
#         num_classes: number of class
#     '''
#     def __init__(self, args):

#         super().__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5) #入力チャンネル数...num_channels
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes) #出力チャンネル数...num_classes

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
class CNNMnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(args.num_channels, 16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.fc = nn.Linear(7 * 7 * 32, args.num_classes)
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
    
class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(modelC, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv9 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv10 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))
        conv9_out = F.relu(self.conv8(conv8_out))
        conv10_out = F.relu(self.conv8(conv9_out))

        class_out = F.relu(self.class_conv(conv10_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
