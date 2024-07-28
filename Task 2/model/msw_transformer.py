import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def to_2tuple(x):
    """
    Ensure that the input x is a tuple.
    If x is not a tuple, return a tuple with x repeated twice.
    """
    if isinstance(x, tuple):
        return x
    return (x, x)


def window_partition(x, window_size):
    """
    Partition the input tensor x into non-overlapping windows of size window_size.

    Args:
        x: Input tensor with shape (B, H, W, C)
        window_size: The size of the windows to partition into

    Returns:
        Tensor with shape (B, num_windows, window_size * window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H // window_size * W // window_size,
               window_size * window_size, C)
    return x


def window_reverse(windows, window_size, H, W):
    """
    Reconstruct the original input tensor from the partitioned windows.

    Args:
        windows: Tensor with partitioned windows of shape (B, num_windows, window_size * window_size, C)
        window_size: The size of the windows
        H: Original height of the input tensor
        W: Original width of the input tensor

    Returns:
        Reconstructed tensor with shape (B, H, W, C)
    """
    B, num_windows, window_size, C = windows.shape
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x


class RelativePositionalBias(nn.Module):
    """
    Module for adding relative positional bias to the attention mechanism.

    Args:
        window_size: The size of the windows
        num_heads: Number of attention heads
    """

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.bias = nn.Parameter(torch.zeros(
            num_heads, window_size * 2 - 1, window_size * 2 - 1))


class Attention(nn.Module):
    """
    Attention mechanism with optional relative positional bias.

    Args:
        dim: Input dimension of the features
        num_heads: Number of attention heads
        dropout: Dropout rate
        window_size: Size of the windows (for relative positional bias)
    """

    def __init__(self, dim, num_heads, dropout=0.0, window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim * \
            num_heads == dim, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if window_size is not None:
            self.relative_position_bias = RelativePositionalBias(
                window_size, num_heads)
        else:
            self.relative_position_bias = None

    def forward(self, x, mask=None):
        """
        Forward pass of the attention mechanism.

        Args:
            x: Input tensor with shape (B, N, C)
            mask: Optional mask tensor

        Returns:
            Output tensor with shape (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        if self.relative_position_bias is not None:
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = attn_weights + self.relative_position_bias.bias
            attn_weights = attn_weights.softmax(dim=-1)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = attn_weights.softmax(dim=-1)

        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MSWFeatureFusion(nn.Module):
    """
    Module for Multi-Scale Window (MSW) feature fusion.

    Args:
        embed_dim: Embedding dimension of the input features
        num_windows: Number of different window sizes to fuse
    """

    def __init__(self, embed_dim, num_windows):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_windows = num_windows

        # Linear projection layer for each window size
        self.proj_layers = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_windows)])
        # Fusion weight layer
        self.fusion_weight = nn.Parameter(torch.ones(num_windows))

    def forward(self, x):
        """
        Combines features from different window sizes using trainable weights.

        Args:
            x: List of tensors, each containing features from a specific window size.

        Returns:
            Fused feature tensor with shape (B, N, C), where:
                - B is the batch size
                - N is the number of patches
                - C is the embedding dimension
        """
        assert len(
            x) == self.num_windows, "Number of inputs doesn't match number of window sizes."

        # Apply linear projection to each window feature
        projected_features = [layer(feat)
                              for layer, feat in zip(self.proj_layers, x)]
        # Apply fusion weights and sum
        weighted_features = [feat * w for feat,
                             w in zip(projected_features, self.fusion_weight)]
        fused_features = torch.sum(torch.stack(weighted_features), dim=0)

        return fused_features


class PatchEmbed(nn.Module):
    """
    ECG Signal to Patch Embedding

    Args:
        signal_length (int): Length of each ECG channel signal. Default: 1000.
        patch_size (int): Patch token size. Default: 5.
        in_chans (int): Number of input channels (leads). Default: 12.
        embed_dim (int): Number of linear projection output channels. Default: 512.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, signal_length=1000, patch_size=5, in_chans=12, embed_dim=512, norm_layer=None):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.num_patches = signal_length // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Forward pass of the patch embedding layer.

        Args:
            x: Input tensor with shape (B, C, L)

        Returns:
            Output tensor with shape (B, num_patches, embed_dim)
        """
        B, C, L = x.shape
        assert L == self.signal_length, f"Input signal length ({L}) doesn't match model ({self.signal_length})."
        x = self.proj(x).transpose(1, 2)  # B, num_patches, embed_dim
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """
        Calculate the number of floating point operations (FLOPs) for the patch embedding layer.

        Returns:
            Number of FLOPs.
        """
        num_patches = self.signal_length // self.patch_size
        flops = num_patches * self.embed_dim * self.in_chans * self.patch_size
        if self.norm is not None:
            flops += num_patches * self.embed_dim
        return flops


class MSWTransformerBlock(nn.Module):
    """
    Multi-Scale Window (MSW) Transformer Block.

    Args:
        embed_dim (int): Embedding dimension of the input features. Default: 512.
        num_heads (int): Number of attention heads. Default: 8.
        window_sizes (list): List of window sizes to use for multi-scale attention. Default: [5, 10, 15].
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.1.
    """

    def __init__(self, embed_dim=512, num_heads=8, window_sizes=[5, 10, 15], mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_sizes = window_sizes
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for window_size in window_sizes:
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(embed_dim),
                    Attention(embed_dim, num_heads, dropout=dropout,
                              window_size=window_size),
                    nn.LayerNorm(embed_dim),
                    nn.Sequential(
                        nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                        nn.Dropout(dropout),
                    )
                ])
            )

        # Feature Fusion module
        self.feature_fusion = MSWFeatureFusion(embed_dim, len(window_sizes))

    def forward(self, x):
        """
        Forward pass of the MSW Transformer block.

        Args:
            x: Input tensor with shape (B, N, C)

        Returns:
            Output tensor with shape (B, N, C)
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)

        # Window partition
        x_windows = []
        for window_size in self.window_sizes:
            x_windows.append(window_partition(x, window_size))

        # Apply MSW-Transformer to each window
        outputs = []
        for i, (window_size, x_window) in enumerate(zip(self.window_sizes, x_windows)):
            num_windows = x_window.shape[1]
            x_window = x_window.view(-1, window_size * window_size, C)
            for layer in self.layers[i]:
                x_window = layer(x_window)
            outputs.append(x_window.view(-1, num_windows,
                           window_size * window_size, C))

        # Flatten and concatenate window outputs
        outputs = [o.view(B, H * W, -1) for o in outputs]
        x = torch.cat(outputs, dim=-1)

        # Perform feature fusion
        x = self.feature_fusion([x[:, i::len(self.window_sizes), :]
                                for i in range(len(self.window_sizes))])

        return x

    def flops(self):
        """
        Calculate the number of floating point operations (FLOPs) for the MSW Transformer block.

        Returns:
            Number of FLOPs.
        """
        flops = 0
        for layer in self.layers:
            ln1, attn, ln2, mlp = layer
            flops += 2 * self.embed_dim * self.num_heads  # LayerNorm FLOPs
            flops += attn.flops()  # Attention FLOPs
            flops += 2 * self.embed_dim * \
                int(self.embed_dim * self.mlp_ratio)  # MLP

        # Add FLOPs for feature fusion
        flops += self.feature_fusion.num_windows * \
            (self.embed_dim + self.embed_dim * len(self.window_sizes))
        return flops
