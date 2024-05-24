import torch.utils.model_zoo as model_zoo
import torch
from torch import nn, Tensor
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',

}

class Downstrteam_channel_attention_model(nn.Module):
    def __init__(self, ch_in,ch_out):
        super(Downstrteam_channel_attention_model, self).__init__()
        self.upsample = nn.ConvTranspose2d(ch_in,ch_out, kernel_size=2,stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_out, ch_out, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.upsample(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return y.expand_as(x)  # 注意力作用每一个通道上

class Upstrteam_channel_attention_model(nn.Module):
    def __init__(self, ch_in,ch_out):
        super(Upstrteam_channel_attention_model, self).__init__()
        self.downsample = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_out, ch_out, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.downsample(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return y.expand_as(x)  # 注意力作用每一个通道上


class Upstrteam_spatial_attention_model(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=7):
        super(Upstrteam_spatial_attention_model, self).__init__()
        self.downsample = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=2)
        padding = 7 // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.downsample(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)

class Downstrteam_spatial_attention_model(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=7):
        super(Downstrteam_spatial_attention_model, self).__init__()
        self.upsample = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        padding = 7 // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.upsample(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)

class Fsuion(nn.Module):
    def __init__(self, ch_in, ch_out, stride=2):
        super(Fsuion, self).__init__()
        self.conv1 = nn.Conv2d(ch_in ,ch_out)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    """残差结构中最后一层卷积核的个数是与之前的4倍"""

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.sge = SpatialGroupEnhance(64)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)

        return out


class DIANet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 look=True):
        super(DIANet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.look = look

        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.DCAM1 = Downstrteam_channel_attention_model(2048,1024)
        self.DCAM2 = Downstrteam_channel_attention_model(1024, 512)
        self.DCAM3 = Downstrteam_channel_attention_model(512, 256)

        self.UCAM1 = Upstrteam_channel_attention_model(256, 512)
        self.UCAM2 = Upstrteam_channel_attention_model(512, 1024)
        self.UCAM3 = Upstrteam_channel_attention_model(1024, 2048)

        self.DSAM1 = Downstrteam_spatial_attention_model(2048,1024)
        self.DSAM2 = Downstrteam_spatial_attention_model(1024, 512)
        self.DSAM3 = Downstrteam_spatial_attention_model(512, 256)

        self.USAM1 = Upstrteam_spatial_attention_model(256, 512)
        self.USAM2 = Upstrteam_spatial_attention_model(512, 1024)
        self.USAM3 = Upstrteam_spatial_attention_model(1024, 2048)

        self.fusion1 = Fsuion(256, 512)
        self.fusion2 = Fsuion(512, 1024)
        self.fusion3 = Fsuion(1024, 2048)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,))

        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        F2 = x       # 56×56×256
        x = self.layer2(x)
        F3 = x       # 28×28×512
        x = self.layer3(x)
        F4 = x       # 14×14×1024
        x = self.layer4(x)
        F5 = x       # 7×7×2048

        C4 = F2
        C5 = F5
        C6 = self.DCAM1(F5) * F4
        C7 = self.DCAM2(C6) * F3
        C8 = self.DCAM3(C7) * F2
        C3 = self.UCAM1(F2) * F3
        C2 = self.UCAM1(C3) * F4
        C1 = self.UCAM1(C2) * F5

        S5 = C5
        S6 = self.DSAM1(C5) * C6
        S7 = self.DSAM2(S6) * C7
        S8 = self.DSAM3(S7) * C8
        S4 = C4
        S3 = self.USAM1(C4) * C3
        S2 = self.USAM2(S3) * C2
        S1 = self.USAM3(S2) * C1

        OUT = self.fusion3(self.fusion2(self.fusion1(S4+S8)+S3+S7)+S2+S6)+S1+S5
        out = self.avg_pool(OUT)
        out = torch.flatten(out, 1)
        # print(out.shape)
        return out


def DIANet_pretained(pretrained=False, num_classes=1000, include_top=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DIANet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                if "fc" not in key and "features.13" not in key:
                    state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


class DIANET(nn.Module):
    # 重新以resnet50为主干构建自己的全连接层
    def __init__(self):
        super(DIANET, self).__init__()
        self.backbone = self._get_backbone()
        self.fc = nn.Linear(2048, 21)

    def _get_backbone(self):
        backbone = DIANet_pretained(pretrained=True, num_classes=config.NUM_CLASSES)
        return backbone

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(-1, 2048)    #   展平为一维
        out = self.fc(out)
        return out

