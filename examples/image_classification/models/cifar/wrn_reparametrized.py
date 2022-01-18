import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['wrn_reparametrized']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, reparametrized=False):
        super(BasicBlock, self).__init__()
        self.register_buffer("reparametrized", torch.tensor(reparametrized))

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        p1 = torch.diag(torch.randn(out_planes))
        # Ensure invertibility
        p1 = (p1 @ p1.T + 1e-2 * torch.eye(out_planes))
        p1 /= torch.norm(p1)
        self.p1 = nn.Parameter(p1, requires_grad=False)
        if reparametrized:
            w1 = self.conv1.weight.data
            wp1 = (w1.permute(1, 2, 3, 0) @ p1.inverse()).permute(3, 0, 1, 2)
            self.conv1_original_weight = nn.Parameter(w1, requires_grad=False)
            self.conv1.weight = nn.Parameter(wp1)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        p2 = torch.diag(torch.randn(out_planes))
        p2 = (p2 @ p2.T + 1e-2 * torch.eye(out_planes))
        p2 /= torch.norm(p2)
        self.p2 = nn.Parameter(p2, requires_grad=False)
        if reparametrized:
            w2 = self.conv2.weight.data
            wp2 = (w2.permute(1, 2, 3, 0) @ p2.inverse()).permute(3, 0, 1, 2)
            self.conv2_original_weight = nn.Parameter(w2, requires_grad=False)
            self.conv2.weight = nn.Parameter(wp2)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.droprate = dropRate


    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        if self.reparametrized:
            out = torch.conv2d(out, self.p1[:, :, None, None])
        
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        if self.reparametrized:
            out = torch.conv2d(out, self.p2[:, :, None, None])

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, reparametrized=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, reparametrized)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, reparametrized):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes, 
                    out_planes, 
                    i == 0 and stride or 1, 
                    dropRate, 
                    reparametrized
                )
            )
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, reparametrized=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, reparametrized)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, reparametrized)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, reparametrized)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn_reparametrized(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model
