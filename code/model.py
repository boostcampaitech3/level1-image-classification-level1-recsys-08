import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class VisionTransformer(nn.Module):
    class ImageEmbedding(nn.Module):
        def __init__(self, num_channel: int, width: int, height: int, patch_size: int,
                     embedded_dim: int, verbose: bool = False):
            super().__init__()
            self.width = width
            self.height = height
            self.verbose = verbose

            self.rearrange = Rearrange('b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c) ', p1=patch_size,
                                       p2=patch_size)
            self.linear = nn.Linear(num_channel * patch_size * patch_size, embedded_dim)

            self.cls_token = nn.Parameter(torch.randn(1, 1, embedded_dim))

            n_patches = self.width * self.height // (patch_size ** 2)
            self.positions = nn.Parameter(torch.randn(n_patches + 1, embedded_dim))

        def forward(self, x):
            batch, channel, width, height = x.shape

            if self.verbose:
                print(x.shape)

            x = self.rearrange(x)

            if self.verbose:
                print(x.shape)

            x = self.linear(x)

            if self.verbose:
                print(x.shape)

            c = repeat(self.cls_token, '() n d -> b n d', b=batch)
            x = torch.cat((c, x), dim=1)
            x = x + self.positions

            return x

    class MultiHeadAttention(nn.Module):
        def __init__(self, embedded_dim: int, num_heads: int, dropout_ratio: float, verbose: bool = False):
            super().__init__()

            self.verbose = verbose
            self.embedded_dim = embedded_dim
            self.num_heads = num_heads
            self.scaling = (embedded_dim // num_heads) ** -0.5

            self.value = nn.Linear(embedded_dim, embedded_dim)
            self.key = nn.Linear(embedded_dim, embedded_dim)
            self.query = nn.Linear(embedded_dim, embedded_dim)
            self.drop = nn.Dropout(dropout_ratio)

            self.linear = nn.Linear(embedded_dim, embedded_dim)

        def forward(self, x: torch.Tensor):
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)

            if self.verbose:
                print(f'Size of Q: {Q.size()}')
                print(f'Size of K: {K.size()}')
                print(f'Size of V: {V.size()}')

            Q = rearrange(Q, 'b q (h d) -> b h q d', h=self.num_heads)
            # K는 Q와 multiplication 해야 하므로 transpose를 취한 형태로 바꿔준다.
            K = rearrange(K, 'b k (h d) -> b h d k', h=self.num_heads)
            V = rearrange(V, 'b v (h d) -> b h v d', h=self.num_heads)

            if self.verbose:
                print(f'Size of Q: {Q.size()}')
                print(f'Size of K: {K.size()}')
                print(f'Size of V: {V.size()}')

            weight = torch.matmul(Q, K)
            weight = weight * self.scaling

            if self.verbose:
                print(f'Size of Weight: {weight.size()}')

            attention = torch.softmax(weight, dim=-1)
            attention = self.drop(attention)

            if self.verbose:
                print(f'Size of Attention: {attention.size()}')

            context = torch.matmul(attention, V)
            context = rearrange(context, 'b h q d -> b q (h d)')

            if self.verbose:
                print(f'Size of Context: {context.size()}')

            x = self.linear(context)
            return x, attention

    class MultiLayerBlock(nn.Module):
        def __init__(self, embedded_dim: int, forward_dim: int, dropout_ratio: float):
            super().__init__()
            self.linear_1 = nn.Linear(embedded_dim, forward_dim * embedded_dim)
            self.dropout = nn.Dropout(dropout_ratio)
            self.linear_2 = nn.Linear(forward_dim * embedded_dim, embedded_dim)

        def forward(self, x):
            x = self.linear_1(x)
            x = nn.functional.gelu(x)
            x = self.dropout(x)
            x = self.linear_2(x)
            x = nn.functional.gelu(x)
            x = self.dropout(x)
            return x

    class EncoderBlock(nn.Sequential):
        def __init__(self, embedded_dim: int, num_heads: int, forward_dim: int, dropout_ratio: float):
            super().__init__()

            self.norm_1 = nn.LayerNorm(embedded_dim)
            self.mha = VisionTransformer.MultiHeadAttention(embedded_dim, num_heads, dropout_ratio)

            self.norm_2 = nn.LayerNorm(embedded_dim)
            self.mlp = VisionTransformer.MultiLayerBlock(embedded_dim, forward_dim, dropout_ratio)

            self.residual_dropout = nn.Dropout(dropout_ratio)

        def forward(self, x):
            x_ = self.norm_1(x)
            x_, attention = self.mha(x_)
            x = x_ + self.residual_dropout(x)

            x_ = self.norm_2(x)
            x_ = self.mlp(x_)
            x = x_ + self.residual_dropout(x)

            return x, attention

    def __init__(self, num_classes: int, num_channel: int = 3, width: int = 96, height: int = 128,
                 patch_size: int = 32, embedded_dim: int = 6 * 6 * 3,
                 num_encoder_layers: int = 3, num_heads: int = 4,
                 forward_dim: int = 4, dropout_ratio: float = 0.2, verbose: bool = False):
        super().__init__()

        self.width = width
        self.height = height
        self.num_channel = num_channel
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embedded_dim = embedded_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout_ratio = dropout_ratio
        self.verbose = verbose

        self.embed_image = self.ImageEmbedding(
            self.num_channel,
            self.width,
            self.height,
            self.patch_size,
            self.embedded_dim,
            self.verbose
        )

        self.transformer_encoders = nn.ModuleList([
            self.EncoderBlock(
                self.embedded_dim,
                self.num_heads,
                self.forward_dim,
                self.dropout_ratio
            ) for _ in range(self.num_encoder_layers)
        ])

        self.reduce_layer = Reduce('b n e -> b e', reduction='mean')
        self.normalization = nn.LayerNorm(self.embedded_dim)
        self.classification_head = nn.Linear(self.embedded_dim, self.num_classes)

    def forward(self, x):
        x = self.embed_image(x)

        attentions = list()
        for encoder in self.transformer_encoders:
            x, attention = encoder(x)
            attentions.append(attention)

        x = self.reduce_layer(x)
        x = self.normalization(x)
        x = self.classification_head(x)

        return x

    # def set_attribute(self, **kwargs):
    #     [setattr(self, key, value) for key, value in kwargs.items()]


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, in_features: int = 512):
        super().__init__()

        self.num_classes = num_classes
        self.in_features = in_features

        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.fc = torch.nn.Linear(in_features=self.in_features, out_features=self.num_classes, bias=True)

        torch.nn.init.kaiming_normal_(self.resnet18.fc.weight)
        stdv = 1. / math.sqrt(self.resnet18.fc.weight.size(1))
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.resnet18(x)
