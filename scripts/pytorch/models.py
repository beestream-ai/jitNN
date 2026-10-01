"""ResNet-18 and MobileNetV2 built exactly like jitnn.nn.models (same layers, same order, same
seeds handed out in construction order), with jitNN's initial weights loaded."""
import torch

from jitnn_init import conv2d, linear


class Seeds:
    def __init__(self, base):
        self.next_seed = base

    def __call__(self):
        s = self.next_seed
        self.next_seed += 1
        return s


def conv(i, o, k, stride, pad, groups, seeds, bias=False):
    m = torch.nn.Conv2d(i, o, k, stride=stride, padding=pad, groups=groups, bias=bias)
    w, b = conv2d(i, o, k, seeds(), groups=groups, bias=bias)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(w))
        if bias:
            m.bias.copy_(torch.from_numpy(b))
    return m


def fc(i, o, seed):
    m = torch.nn.Linear(i, o)
    w, b = linear(i, o, seed)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(w))
        m.bias.copy_(torch.from_numpy(b))
    return m


class Add(torch.nn.Module):
    def __init__(self, main, shortcut, relu):
        super().__init__()
        self.main, self.shortcut, self.relu = main, shortcut, relu

    def forward(self, x):
        y = self.main(x) + (x if self.shortcut is None else self.shortcut(x))
        return torch.relu(y) if self.relu else y


def basic_block(i, o, stride, seeds):
    main = torch.nn.Sequential(conv(i, o, 3, stride, 1, 1, seeds), torch.nn.BatchNorm2d(o), torch.nn.ReLU(),
                               conv(o, o, 3, 1, 1, 1, seeds), torch.nn.BatchNorm2d(o))
    short = torch.nn.Sequential(conv(i, o, 1, stride, 0, 1, seeds), torch.nn.BatchNorm2d(o)) if stride != 1 or i != o else None
    return Add(main, short, True)


def resnet18(in_channels, classes, small, seed):
    seeds = Seeds(seed)
    layers = []
    if small:
        layers += [conv(in_channels, 64, 3, 1, 1, 1, seeds), torch.nn.BatchNorm2d(64), torch.nn.ReLU()]
    else:
        layers += [conv(in_channels, 64, 7, 2, 3, 1, seeds), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(3, 2, 1)]
    i = 64
    for stage, w in enumerate([64, 128, 256, 512]):
        for b in range(2):
            layers.append(basic_block(i, w, 2 if stage > 0 and b == 0 else 1, seeds))
            i = w
    layers += [torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), fc(512, classes, seeds())]
    return torch.nn.Sequential(*layers)


SETTINGS = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


def inverted_residual(i, o, stride, t, seeds):
    hidden = i * t
    m = []
    if t != 1:
        m += [conv(i, hidden, 1, 1, 0, 1, seeds), torch.nn.BatchNorm2d(hidden), torch.nn.ReLU6()]
    m += [conv(hidden, hidden, 3, stride, 1, hidden, seeds), torch.nn.BatchNorm2d(hidden), torch.nn.ReLU6(),
          conv(hidden, o, 1, 1, 0, 1, seeds), torch.nn.BatchNorm2d(o)]
    main = torch.nn.Sequential(*m)
    return Add(main, None, False) if stride == 1 and i == o else main


def mobilenet_v2(in_channels, classes, small, seed):
    seeds = Seeds(seed)
    layers = [conv(in_channels, 32, 3, 1 if small else 2, 1, 1, seeds), torch.nn.BatchNorm2d(32), torch.nn.ReLU6()]
    i = 32
    for idx, (t, c, n, s) in enumerate(SETTINGS):
        for r in range(n):
            stride = s if r == 0 else 1
            if small and idx == 1:
                stride = 1
            layers.append(inverted_residual(i, c, stride, t, seeds))
            i = c
    layers += [conv(i, 1280, 1, 1, 0, 1, seeds), torch.nn.BatchNorm2d(1280), torch.nn.ReLU6(),
               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), fc(1280, classes, seeds())]
    return torch.nn.Sequential(*layers)
