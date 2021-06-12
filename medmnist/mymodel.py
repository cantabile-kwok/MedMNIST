import torch.nn as nn
import torch.nn.functional as F

class ResBlock18(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock18, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class ResBlock50(nn.Module):
    def __init__(self, in_channels, tmp_channels, stride=1):
        super(ResBlock50, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, tmp_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(tmp_channels, tmp_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(tmp_channels)
        self.conv3 = nn.Conv2d(tmp_channels, self.expansion*tmp_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(tmp_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*tmp_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * tmp_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * tmp_channels)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.tmp_channels = 64
        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_block(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_block(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_block(block, 512, num_blocks[3], stride=2)
        if num_blocks[0] == 2:
            self.linear = nn.Linear(512, num_classes)
        else:
            self.linear = nn.Linear(2048, num_classes)

    def make_block(self, block, in_channels, num_blocks, stride):
        stride_list = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in stride_list:
            layers.append(block(self.tmp_channels, in_channels, stride))
            if block == ResBlock50:
                self.tmp_channels = in_channels * 4
            else:
                self.tmp_channels = in_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(in_channels, num_classes):
    return ResNet(ResBlock18, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


def ResNet50(in_channels, num_classes):
    return ResNet(ResBlock50, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


if __name__ == '__main__':
    A = ResNet50(3, 2)
    print(A)
