import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch.backends.cuda import sdp_kernel, SDPBackend

from utils import set_seed
from constants import *

backend_map = {
    SDPBackend.MATH: {
        "enable_math": True,
        "enable_flash": False,
        "enable_mem_efficient": False,
    },
    SDPBackend.FLASH_ATTENTION: {
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": False,
    },
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False,
        "enable_flash": False,
        "enable_mem_efficient": True,
    },
}


#######################################


class Net(nn.Module):
    """
    Standard convolutional neural network for image classification.
    """

    def __init__(self, return_act=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm2d_1 = nn.BatchNorm2d(128)
        self.batchnorm2d_2 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 100)  # 100 classes for fine labels

        self.return_act = return_act

    def forward(self, x):

        x = self.conv1(x)
        act1 = x  # (BATCH_SIZE, 64, 30, 30)
        x = F.relu(x)

        x = self.conv2(x)
        act2 = x  # (BATCH_SIZE, 128, 28, 28)
        x = F.relu(x)
        x = self.batchnorm2d_1(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        act3 = x  # (BATCH_SIZE, 256, 12, 12)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        act4 = x  # (BATCH_SIZE, 512, 4, 4)
        x = F.relu(x)
        x = self.batchnorm2d_2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.return_act:
            return x, (act1, act2, act3, act4)
        else:
            return x


def get_model_and_optimizer(seed=None, student=False, ct_model=None, return_act=False):
    """
    student (bool=False): If True, indicates instantiation of a student model to be unlearned via STUDENT_LR
    return_act (bool=False): If True, instantiated model will return activation after every conv layer
    """
    if student:
        assert (
            ct_model is not None
        ), "If initializing a student model, must pass in a competent teacher `ct_model`."

    if seed:
        set_seed(seed)
    model = Net(return_act=return_act)
    if student:
        model.load_state_dict(ct_model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=STUDENT_LR)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    return model, optimizer


class AttackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)  # input = shadow model probs for CIFAR-100
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_attack_model_and_optimizer():
    model = AttackNet()
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


#############
# Vision Transformer


class MLP(nn.Module):
    def __init__(self, n_embd, n_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_kv_head,
        device,
        dropout=0.1,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.drop = nn.Dropout(p=dropout)

        self.n_kv_head = n_kv_head
        self.n_repeat = self.n_head // self.n_kv_head

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.value = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)

        self.device = device

    def split_heads(self, x, n_head):
        B, S, D = x.size()
        # split dimension into n_head * head_dim, then transpose the sequence length w/ n_head
        # output: [B, n_head, S, head_dim]
        return x.view(B, S, n_head, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, _, S, head_dim = x.size()  # _ is n_head which we will merge
        # output: [B, S, n_embd]
        return x.transpose(1, 2).contiguous().view(B, S, self.n_embd)

    def forward(self, x):
        # x: (B, S, n_embd)
        # Step 1 and 2: Project query, key, value, then split via reshaping
        q = self.split_heads(self.query(x), self.n_head)
        k = self.split_heads(self.key(x), self.n_kv_head)
        v = self.split_heads(self.value(x), self.n_kv_head)

        ## GQA
        k, v = repeat_kv(k, v, self.n_repeat)
        assert (
            k.shape[1] == self.n_head and v.shape[1] == self.n_head
        ), "key and value n_head do not match query n_head"
        # q, k, v [B, n_head, S, head_dim)

        # Step 3: Compute scaled dot-product attention
        with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
            try:
                attn = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.drop.p if self.device.type == "cuda" else 0
                )  # ViT: not causal ofc
            # CPU: Both fused kernels do not support non-zero dropout. (Dec 2023)
            except RuntimeError:
                print("FlashAttention is not supported. See warnings for reasons.")

        # Step 4 and 5: Concatenate attention scores, return projected output matrix
        out = self.out(self.combine_heads(attn))  # (B, S, n_embd)
        return out


# helper function for GQA
def repeat_kv(k, v, n_repeat):
    k = torch.repeat_interleave(k, repeats=n_repeat, dim=1)
    v = torch.repeat_interleave(v, repeats=n_repeat, dim=1)
    return k, v


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_ff, n_kv_head, device, norm_first, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd,
            n_head,
            n_kv_head,
            device,
            dropout,
        )
        self.ff = MLP(n_embd, n_ff, dropout=dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.norm_first = norm_first
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # residual connection (stream)

        # pre layer norm
        if self.norm_first:
            x = x + self.drop(self.sa(self.ln1(x)))
            x = x + self.drop(self.ff(self.ln2(x)))
        else:
            x = self.ln1(x + self.drop(self.sa(x)))
            x = self.ln2(x + self.drop(self.ff(x)))

        return x


###########################################


class PatchEmbedding(nn.Module):
    """
    Applies patch embeddings to an image.
    """

    def __init__(self, patch_size, n_embd, in_channels=3):
        super().__init__()

        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels, n_embd, kernel_size=patch_size, stride=patch_size
        )
        # self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # (B, C, img_size, img_size) -> (B, num_patches, n_embd)

        x = self.conv(x)  # (B, n_embd, img_size//patch_size, img_size//patch_size)
        x = rearrange(x, "b c h w -> b (h w) c")
        # equivalent to above line: x = x.flatten(2).transpose(-1, -2)
        return x


class ViT(nn.Module):
    """
    Standard Vision Transformer.

    TODO
    img_size (int): Width/height of input images (assuming square), e.g. 32 for CIFAR-10
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_ff,
        n_layer,
        n_class,
        img_size,
        patch_size,
        device,
        norm_first,
        n_kv_head=None,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size, n_embd)
        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches, n_embd))

        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, n_ff, n_kv_head, device, norm_first, dropout)
                for i in range(n_layer)
            ]
        )

        self.mlp_head = nn.Linear(n_embd, n_class)
        self.drop = nn.Dropout(dropout)
        self.device = device
        self.init_params()

    # weight initialization (Xavier uniform)
    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    # excludes layer norm and biases
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

    # Remark: Xavier normal is not supported at this time.

    def forward(self, x):
        B = x.shape[0]  # (B, num_channels, img_size, img_size)

        x = self.patch_embedding(x)

        cls_token = self.cls_token.expand(
            B, -1, -1
        )  # prepares cls_token from (1,1,n_embd) to (B, 1, n_embd)

        x, ps = pack(
            (cls_token, x), "b * n_embd"
        )  # expand into dim=1 to form num_patches+1
        # identical to x = torch.cat((cls_token, x), dim=1)

        x += self.pos_embedding  # identical shape, no broadcasting
        x = self.drop(x)
        # (B, 1+num_patches, n_embd)

        for block in self.blocks:
            x = block(x)  # (B, 1+num_patches, n_embd)

        # retrieve logits on the class token only
        x, _ = unpack(x, ps, "b * n_embd")
        x = rearrange(x, "b 1 n_embd -> b n_embd")
        logits = self.mlp_head(x)  # (B, n_class)

        return logits


def get_vit_and_optimizer(seed=None, **kwargs):
    if seed:
        set_seed(seed)
    model = ViT(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    return model, optimizer
