"""
===============================================================================
========================You can run this script directly=======================
===============================================================================

This is the loss function we changed for the stable diffusion
For more detail about the design, please refer to the report

You can get the output like this:
    batch size: 8
    input image shape: torch.Size([8, 3, 32, 32])
    input time shape: torch.Size([8])
    input label shape: torch.Size([8, 3, 32, 32])
    output shape: torch.Size([8, 3, 32, 32])

===============================================================================
===============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ImageFlatten(nn.Module):
    def __init__(self, image_size=32, output_dim=512):
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(image_size * image_size * 3, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb


def sinusoidal_positional_encoding(d_model, num_patches):
    pe = torch.zeros(num_patches, d_model)
    position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


class CustomImageEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=8, in_chans=3, embed_dim=512, num_heads=8, num_layers=6, output_dim=512):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.pe = sinusoidal_positional_encoding(embed_dim, num_patches)

        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.patch_embed(x)

        x = x + self.pe.to(x.device)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc_out(x)

        # Normalize the output vector to have a length of 1
        # x = F.normalize(x, p=2, dim=-1)

        # Apply Sigmoid function to limit the output range to [0, 1]
        x = torch.sigmoid(x)

        # print(x.min(), x.max(), x.mean())

        return x


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


####################################################################################################
# the resizer model to resize the size of image prompts
class Resizer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Resizer, self).__init__()
        in_channels = input_shape[1]
        out_channels = output_shape[1]
        out_size = output_shape[2]
        height = input_shape[2]

        num_operations = int(abs(numpy.log2(height / out_size)))

        # assemble the layers
        if out_size > height:
            self.model = nn.Sequential(
                *[nn.ConvTranspose2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=4, stride=2, padding=1) for i in range(num_operations)]
            )
        elif out_size < height:
            self.model = nn.Sequential(
                *[nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=2, padding=1) for i in range(num_operations)]
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        return self.model(x)


####################################################################################################

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )

        ####################################################################################################

        self.resize_model_128_32_to_128_16 = Resizer([128, 32, 32], [128, 16, 16])
        self.resize_model_128_16_to_256_16 = Resizer([128, 16, 16], [256, 16, 16])
        self.resize_model_256_16_to_256_8 = Resizer([256, 16, 16], [256, 8, 8])
        self.resize_model_256_8_to_256_4 = Resizer([256, 8, 8], [256, 4, 4])
        self.resize_model_256_4_to_512_4 = Resizer([256, 4, 4], [512, 4, 4])
        self.resize_model_512_4_to_512_8 = Resizer([512, 4, 4], [512, 8, 8])
        self.resize_model_512_8_to_512_16 = Resizer([512, 8, 8], [512, 16, 16])
        self.resize_model_512_16_to_384_16 = Resizer([512, 16, 16], [384, 16, 16])
        self.resize_model_384_16_to_384_32 = Resizer([384, 16, 16], [384, 32, 32])
        self.resize_model_384_32_to_256_32 = Resizer([384, 32, 32], [256, 32, 32])
        self.resize_model_256_32_to_3_32 = Resizer([256, 32, 32], [3, 32, 32])

        ####################################################################################################

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb, labels):

        # print(x.shape)

        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]  # batch size , 512 , 1 , 1

        now_channel = labels.shape[1]
        now_size = labels.shape[2]

        target_channel = h.shape[1]
        target_size = h.shape[2]

        """
        self.resize_model_128_32_to_128_16 = Resizer([128, 32, 32], [128, 16, 16])
        self.resize_model_128_16_to_256_16 = Resizer([128, 16, 16], [256, 16, 16])
        self.resize_model_256_16_to_256_8 = Resizer([256, 16, 16], [256, 8, 8])
        self.resize_model_256_8_to_256_4 = Resizer([256, 8, 8], [256, 4, 4])
        self.resize_model_256_4_to_512_4 = Resizer([256, 4, 4], [512, 4, 4])
        self.resize_model_512_4_to_512_8 = Resizer([512, 4, 4], [512, 8, 8])
        self.resize_model_512_8_to_512_16 = Resizer([512, 8, 8], [512, 16, 16])
        self.resize_model_512_16_to_384_16 = Resizer([512, 16, 16], [384, 16, 16])
        self.resize_model_384_16_to_384_32 = Resizer([384, 16, 16], [384, 32, 32])
        self.resize_model_384_32_to_256_32 = Resizer([384, 32, 32], [256, 32, 32])
        self.resize_model_256_32_to_3_32 = Resizer([256, 32, 32], [3, 32, 32])
        """

        # the scheduler of the resizer model
        # add the output to hidden layer
        ####################################################################################################
        if now_channel == 128 and now_size == 32 and target_channel == 128 and target_size == 16:
            h += self.resize_model_128_32_to_128_16(labels)

        if now_channel == 128 and now_size == 16 and target_channel == 256 and target_size == 16:
            h += self.resize_model_128_16_to_256_16(labels)

        if now_channel == 256 and now_size == 16 and target_channel == 256 and target_size == 8:
            h += self.resize_model_256_16_to_256_8(labels)

        if now_channel == 256 and now_size == 8 and target_channel == 256 and target_size == 4:
            h += self.resize_model_256_8_to_256_4(labels)

        if now_channel == 256 and now_size == 4 and target_channel == 512 and target_size == 4:
            h += self.resize_model_256_4_to_512_4(labels)

        if now_channel == 512 and now_size == 4 and target_channel == 512 and target_size == 8:
            h += self.resize_model_512_4_to_512_8(labels)

        if now_channel == 512 and now_size == 8 and target_channel == 512 and target_size == 16:
            h += self.resize_model_512_8_to_512_16(labels)

        if now_channel == 512 and now_size == 16 and target_channel == 384 and target_size == 16:
            h += self.resize_model_512_16_to_384_16(labels)

        if now_channel == 384 and now_size == 16 and target_channel == 384 and target_size == 32:
            h += self.resize_model_384_16_to_384_32(labels)

        if now_channel == 384 and now_size == 32 and target_channel == 256 and target_size == 32:
            h += self.resize_model_384_32_to_256_32(labels)

        if now_channel == 256 and now_size == 32 and target_channel == 3 and target_size == 32:
            h += self.resize_model_256_32_to_3_32(labels)

        ####################################################################################################

        # resize_model.to(x.device)
        # resize_model.eval()

        # h += self.cond_proj(labels)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)  # 512
        self.cond_embedding = CustomImageEmbedding(output_dim=tdim)  # 512
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, t, labels):
        ####################################################################################################
        # Timestep embedding
        temb = self.time_embedding(t)
        # cemb = self.cond_embedding(labels)

        ####################################################################################################

        # use the same process like x
        cemb = self.head(labels)

        # Downsampling
        h = self.head(x)

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randn(batch_size, 3, 32, 32)
    y = model(x, t, labels)

    print(f"batch size: {batch_size}")
    print(f"input image shape: {x.shape}")
    print(f"input time shape: {t.shape}")
    print(f"input label shape: {labels.shape}")
    print(f"output shape: {y.shape}")
