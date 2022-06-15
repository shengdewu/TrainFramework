import torch


class ChannelAttentionModule(torch.nn.Module):
    def __init__(self, ic, ratio):
        super(ChannelAttentionModule, self).__init__()
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.max = torch.nn.AdaptiveMaxPool2d(1)

        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(ic, ic // ratio, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ic // ratio, ic, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.sigmoid = torch.nn.Sigmoid()

        return

    def forward(self, x):
        max_pool = self.max(x)
        avg_pool = self.avg(x)

        max_mlp = self.mlp(max_pool)
        avg_mlp = self.mlp(avg_pool)

        return self.sigmoid(torch.add(max_mlp, avg_mlp))


class SpatialAttentionModule(torch.nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        return

    def forward(self, x):
        max_pool, max_index = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        agg = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(agg)
        return self.sigmoid(conv)


class ConvBlockAttentionModule(torch.nn.Module):
    def __init__(self, ic, ratio=16):
        super(ConvBlockAttentionModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(ic, ratio)
        self.spatial_attention = SpatialAttentionModule()
        return

    def forward(self, x):
        ca = self.channel_attention(x)
        f = torch.mul(ca, x)
        sa = self.spatial_attention(f)
        return torch.mul(sa, f)


class DualAttentionFeatureModule(torch.nn.Module):
    def __init__(self, ic):
        super(DualAttentionFeatureModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(ic, 1)
        self.spatial_attention = SpatialAttentionModule()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(ic, ic, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.PReLU(),
            torch.nn.Conv2d(ic, ic, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.conv1 = torch.nn.Conv2d(ic*2, ic, kernel_size=1, padding=0, stride=1, bias=False)
        return

    def forward(self, x):
        head = self.head(x)
        sa = torch.mul(self.spatial_attention(head), head)
        ca = torch.mul(self.channel_attention(head), head)
        action = self.conv1(torch.cat([sa, ca], dim=1))
        return torch.add(x, action)
