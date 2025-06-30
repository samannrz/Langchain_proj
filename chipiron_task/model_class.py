# Model class
import torch
from torch import nn


class ConvLayer(nn.Module):
    """A single convolutional layer with ReLU activation: in -> conv -> relu -> out.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of the convolutional kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialize the conv layer."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of conv layer."""
        return self.relu(self.conv(x))


class DenseLayer(nn.Module):
    """A convolutional layer with ReLU activation that concatenates its output with the input.

    Structure:in -> conv -> relu -> concat -> out
    â•°â”€----------------------â•¯
    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of the convolutional kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialize the dense layer."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of dense layer."""
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DenseBlock(nn.Module):
    """A block of multiple DenseLayers, sequentially applying convolution and concatenation.

    Args:
        in_channels: Input channels.
        growth_rate: Channel growth per layer.
        n_layers: Number of DenseLayers in the block.
    """

    def __init__(self, in_channels: int, growth_rate: int, n_layers: int) -> None:
        """Initialize the dense block with n_layers."""
        super().__init__()
        self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
        for i in range(n_layers - 1):
            self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size=3))
        self.block = nn.Sequential(*self.block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of dense block."""
        return torch.cat([x, self.block(x)], 1)


class SRDenseNet2D(nn.Module):
    """A 2D Super-Resolution DenseNet for enhancing image resolution.

    Args:
        n_slices: Number of input image slices (channels).
        growth_rate: Growth rate for DenseBlock outputs.
        sr_factor: Super-resolution factor (e.g., 2x or 4x upsampling).
        n_blocks: Number of DenseBlocks.
        n_layers: Number of DenseLayers per DenseBlock.
        n_inter_channels: Number of intermediate channels in the bottleneck layer.
    """

    def __init__(
        self,
        n_slices: int = 18,
        growth_rate: int = 16,
        sr_factor: int = 2,
        n_blocks: int = 8,
        n_layers: int = 8,
        n_inter_channels: int = 256,
    ) -> None:
        """Initialize the model."""
        super().__init__()

        # Initial convolution layer
        self.conv = ConvLayer(
            in_channels=n_slices, out_channels=growth_rate * n_slices, kernel_size=3
        )

        # Dense blocks
        current_channels = growth_rate * n_slices
        self.dense_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.dense_blocks.append(DenseBlock(current_channels, growth_rate, n_layers))
            current_channels += growth_rate * n_layers

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, n_inter_channels, kernel_size=1), nn.ReLU(inplace=True)
        )

        # Deconvolution layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                n_inter_channels,
                n_inter_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True)
        )

        # Reconstruction layer
        self.reconstruction = nn.Conv2d(n_inter_channels, n_slices, kernel_size=3, padding=1)

        self._initialize_weights()


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.conv(x)
        for block in self.dense_blocks:
            x = block(x)
        x = self.bottleneck(x)
        x = self.deconv(x)
        x = self.reconstruction(x)
        return x
