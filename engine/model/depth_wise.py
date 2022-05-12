import torch


class DepthWiseSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DepthWiseSeparableConv2d, self).__init__()

        self.depth = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=in_channels,
                                     dilation=dilation,
                                     bias=bias)
        self.wise = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=bias)
        return

    def forward(self, x):
        x = self.depth(x)
        return self.wise(x)

    def apply(self, init_fn):
        self.depth.apply(init_fn)
        self.wise.apply(init_fn)
        return
