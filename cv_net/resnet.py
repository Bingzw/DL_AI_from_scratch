import torch.nn as nn
from types import SimpleNamespace
from cv_net.util.util import act_fn_by_name


class ResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        :param c_in: number of input channels
        :param act_fn: activation function
        :param subsample: whether to apply downsampling
        :param c_out: number of output channels
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # network representing F
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2 if subsample else 1, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        self.down_sample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.down_sample is not None:
            x = self.down_sample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class PreActResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        :param c_in: Number of input channels
        :param act_fn: Activation class constructor (e.g. nn.ReLU)
        :param subsample: If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
        :param c_out: Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        )

        # 1x1 convolution can apply non-linearity as well, but not strictly necessary
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False)
        ) if subsample else None

    def forward(self, x):
        z = self.net(x)
        if self.down_sample is not None:
            x = self.down_sample(x)
        out = z + x
        return out


resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock
}


class ResNet(nn.Module):
    def __init__(self, num_classes=10, num_blocks=[3,3,3], c_hidden=[12,32,64], act_fn_name="relu",
                 block_name="ResNetBlock", **kwargs):
        """
        :param num_classes: Number of classes in the dataset
        :param num_blocks: List with the number of ResNet blocks to use. The first block of each group uses
                        downsampling, except the first.
        :param c_hidden: List with the hidden dimensionalities in the different blocks.
                        Usually multiplied by 2 the deeper we go.
        :param act_fn_name: Name of the activation function
        :param block_name: Name of the block to use
        """
        super().__init__()
        assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       num_blocks=num_blocks,
                                       c_hidden=c_hidden,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name],
                                       block_class=resnet_blocks_by_name[block_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden

        # first convolution on the original image to scale up the channel size
        if self.hparams.block_class == PreActResNetBlock: # don't apply non-linearity before the first block
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False)
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                self.hparams.act_fn()
            )

        # create the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0)
                c_in = c_hidden[block_idx - 1] if subsample else c_hidden[block_idx]
                blocks.append(self.hparams.block_class(c_in, self.hparams.act_fn, subsample, c_hidden[block_idx]))
        self.blocks = nn.Sequential(*blocks)

        # map to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x






