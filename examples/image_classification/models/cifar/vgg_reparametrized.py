'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class ReparametrizedConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding=1, reparametrized=False
    ):
        super(ReparametrizedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.register_buffer("reparametrized", torch.tensor(reparametrized))

        n = kernel_size**2 * out_channels
        w = torch.normal(
            0., 
            math.sqrt(2. / n), 
            [out_channels, in_channels, kernel_size, kernel_size]
        )
        self.w = nn.Parameter(w, requires_grad=True)

        p = torch.diag(torch.randn(out_channels))
        # Ensure invertibility
        p = (p @ p.T + 1e-2 * torch.eye(out_channels))
        p /= torch.norm(p)
        self.p = nn.Parameter(p, requires_grad=False)

        wp = (w.permute(1, 2, 3, 0) @ p.inverse()).permute(3, 0, 1, 2)
        self.wp = nn.Parameter(wp, requires_grad=True)

        self.b = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

    def forward(self, x, reparametrized=None):
        if reparametrized is None:
            reparametrized = self.reparametrized
        else:
            print(
                f"Overriding module's state (reparametrized={self.reparametrized}) "
                f"with reparametrized={reparametrized}"
            )
        if reparametrized:
            out = torch.conv2d(x, self.wp, padding=self.padding)
            out = torch.conv2d(out, self.p[:, :, None, None])
        else:
            out = torch.conv2d(x, self.w, padding=self.padding)

        out += self.b[None, :, None, None]
        return out

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, reparametrized=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.reparametrized(reparametrized)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reparametrized(self, val):
        for m in self.modules():
            if isinstance(m, ReparametrizedConv2d):
                m.reparametrized = torch.tensor(val)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, reparametrized=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = ReparametrizedConv2d(
                in_channels, v, kernel_size=3, padding=1, reparametrized=reparametrized
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(reparametrized=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], reparametrized=reparametrized), **kwargs)
    return model


def vgg11_bn(reparametrized=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True, reparametrized=reparametrized), **kwargs)
    return model


def vgg13(reparametrized=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], reparametrized=reparametrized), **kwargs)
    return model


def vgg13_bn(reparametrized=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True, reparametrized=reparametrized), **kwargs)
    return model


def vgg16(reparametrized=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], reparametrized=reparametrized), **kwargs)
    return model


def vgg16_bn(reparametrized=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True, reparametrized=reparametrized), **kwargs)
    return model


def vgg19(reparametrized=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], reparametrized=reparametrized), **kwargs)
    return model


def vgg19_bn(reparametrized=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True, reparametrized=reparametrized), **kwargs)
    return model
