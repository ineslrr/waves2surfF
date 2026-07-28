"""Neural network used to map spatial ocean fields to surface velocity.

The U-Net extracts features at several spatial scales and restores the input
resolution through a decoder with skip connections. Global scalar metadata can
be ignored, appended as input channels, compressed to four channels, or used
to modulate intermediate features with FiLM.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _normalization(kind: str, channels: int) -> nn.Module:
    """Construct the requested feature normalization layer.

    GroupNorm is the default because it normalizes each sample independently
    and remains stable when high-resolution fields force small GPU batches.
    BatchNorm can be useful with large batches, while ``none`` supports an
    explicit ablation.
    """
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        # GroupNorm requires the channel count to be divisible by the number
        # of groups. Choose the largest valid value no greater than eight.
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown normalization {kind!r}")


class DoubleConv(nn.Module):
    """Two local feature-extraction layers used throughout the U-Net.

    A stride of two on the first convolution downsamples encoder features.
    Decoder blocks use stride one and therefore preserve spatial resolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        normalization: str = "group",
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            _normalization(normalization, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _normalization(normalization, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a batch shaped ``[B,C,H,W]``."""
        return self.layers(x)


class Up(nn.Module):
    """Upsample a decoder feature and merge it with an encoder skip feature.

    The skip connection restores fine spatial detail that would otherwise be
    lost at the coarse U-Net bottleneck.
    """
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, normalization: str
    ) -> None:
        super().__init__()
        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Return a fused feature map at the skip connection's resolution."""
        # Explicitly target the skip size, which also handles odd image sizes.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class FeatureModulation(nn.Module):
    """Residual FiLM: ``x * (1 + gamma(z)) + beta(z)``.

    One metadata vector produces a scale and offset for every feature channel.
    These values are broadcast over space, allowing global conditions such as
    season or mean temperature to change how spatial features are interpreted.
    """

    def __init__(self, metadata_width: int, channels: int) -> None:
        super().__init__()
        self.generator = nn.Linear(metadata_width, 2 * channels)
        # Zero initialization gives gamma=beta=0 initially, so adding FiLM does
        # not perturb the ordinary U-Net at the start of training.
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Modulate ``x[B,C,H,W]`` using encoded metadata ``[B,D]``."""
        gamma, beta = self.generator(metadata).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class MetadataMLP(nn.Module):
    """Learn a compact nonlinear representation of global metadata.

    The same small multilayer perceptron supports the four-channel broadcast
    experiment and produces the shared embedding consumed by FiLM layers.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """Encode a batch of metadata vectors shaped ``[B,D]``."""
        return self.layers(metadata)


class Waves2SurfNet(nn.Module):
    """Waves2surfF model mapping ``[B, C_in, H, W]`` to ``[B, C_out, H, W]``.

    The backbone is a U-Net with a configurable number of stride-2 downsampling
    stages (``depth``). The default ``depth=4`` matches the original four-level
    network (resolution ÷16 at the bottleneck). ``base_channels=8`` is the
    recommended lightweight default. Group normalization is robust to the small
    per-device batches common in high-resolution geophysical training.

    Conditioning modes:
        ``none`` ignores metadata.
        ``extra_channels`` broadcasts every raw metadata value over the grid.
        ``broadcast`` first compresses metadata to a fixed number of channels.
        ``film`` modulates features throughout encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        base_channels: int = 8,
        normalization: str = "group",
        metadata_dim: int = 0,
        conditioning: str = "none",
        metadata_channels: int = 4,
        metadata_width: int = 32,
        depth: int = 4,
    ) -> None:
        """Build the full network and the selected conditioning pathway.

        Args:
            depth: Number of stride-2 stages (encoder downs + bottleneck).
                ``depth=4`` is the historical default; ``depth=2`` is a
                shallower U-Net (resolution ÷4 at the bottleneck).
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"model depth must be >= 1, got {depth}")
        valid_conditioning = {"none", "extra_channels", "broadcast", "film"}
        if conditioning not in valid_conditioning:
            raise ValueError(
                f"Unknown conditioning {conditioning!r}; expected {valid_conditioning}"
            )
        if conditioning != "none" and metadata_dim < 1:
            raise ValueError(f"{conditioning!r} conditioning requires metadata_dim > 0")
        self.conditioning = conditioning
        self.metadata_dim = metadata_dim
        self.metadata_channels = metadata_channels
        self.depth = depth

        # Input-channel approaches change only the first convolution's width.
        # FiLM instead keeps spatial inputs unchanged and creates an embedding
        # used by lightweight modulation layers.
        if conditioning == "extra_channels":
            stem_channels = in_channels + metadata_dim
            self.metadata_encoder = nn.Identity()
        elif conditioning == "broadcast":
            stem_channels = in_channels + metadata_channels
            self.metadata_encoder = MetadataMLP(
                metadata_dim, metadata_width, metadata_channels
            )
        else:
            stem_channels = in_channels
            self.metadata_encoder = (
                MetadataMLP(metadata_dim, metadata_width, metadata_width)
                if conditioning == "film"
                else None
            )

        # Deeper levels use more channels because their smaller feature maps
        # can represent richer large-scale structure at moderate memory cost.
        # ``widths`` has one entry per skip level (stem + intermediate downs).
        widths = [base_channels * 2**i for i in range(depth)]
        bottleneck_channels = widths[-1] * 2
        self.stem = DoubleConv(
            stem_channels, widths[0], normalization=normalization
        )
        # Intermediate encoder downs (stride 2). The final coarsening is the
        # bottleneck below, so together they provide ``depth`` downsamplings.
        self.downs = nn.ModuleList(
            DoubleConv(
                widths[i], widths[i + 1], stride=2, normalization=normalization
            )
            for i in range(depth - 1)
        )
        self.bottleneck = DoubleConv(
            widths[-1], bottleneck_channels, stride=2, normalization=normalization
        )
        # Decoder: first fuse bottleneck with the coarsest skip, then walk back
        # toward full resolution.
        ups: list[nn.Module] = [
            Up(bottleneck_channels, widths[-1], widths[-1], normalization)
        ]
        for i in range(depth - 2, -1, -1):
            ups.append(Up(widths[i + 1], widths[i], widths[i], normalization))
        self.ups = nn.ModuleList(ups)
        # A 1x1 convolution maps learned features to u/v without mixing nearby
        # pixels or imposing an output activation/range.
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        if conditioning == "film":
            # FiLM on: stem, each down output, bottleneck, each up output.
            film_widths = [*widths, bottleneck_channels, *widths[::-1]]
            self.film = nn.ModuleList(
                FeatureModulation(metadata_width, channels)
                for channels in film_widths
            )
        else:
            self.film = None

    @staticmethod
    def _broadcast(metadata: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand ``[B,D]`` metadata into constant ``[B,D,H,W]`` maps."""
        return metadata[:, :, None, None].expand(
            -1, -1, x.shape[-2], x.shape[-1]
        )

    def _modulate(
        self, index: int, x: torch.Tensor, metadata: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply the indexed FiLM layer, or act as identity in other modes."""
        if self.film is None:
            return x
        return self.film[index](x, metadata)

    def forward(
        self, x: torch.Tensor, metadata: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict velocity from spatial fields and optional metadata.

        Args:
            x: Normalized spatial inputs ``[batch, input_channels, height, width]``.
            metadata: Normalized global values ``[batch, metadata_dim]``.

        Returns:
            Dense output fields ``[batch, output_channels, height, width]``.
        """
        # Both channel-based modes concatenate metadata before the U-Net stem.
        if self.conditioning in {"extra_channels", "broadcast"}:
            if metadata is None:
                raise ValueError("Metadata tensor is required by the configured model")
            encoded = self.metadata_encoder(metadata)
            x = torch.cat((x, self._broadcast(encoded, x)), dim=1)
        elif self.conditioning == "film":
            if metadata is None:
                raise ValueError("Metadata tensor is required by the configured model")
            metadata = self.metadata_encoder(metadata)

        # Encoder: progressively reduce resolution to grow the receptive field.
        # ``skips`` retains stem + intermediate features for the decoder.
        skips: list[torch.Tensor] = []
        h = self._modulate(0, self.stem(x), metadata)
        skips.append(h)
        for i, down in enumerate(self.downs):
            h = self._modulate(i + 1, down(h), metadata)
            skips.append(h)
        h = self._modulate(self.depth, self.bottleneck(h), metadata)
        # Decoder: recover resolution while combining coarse context with the
        # matching fine-resolution encoder features.
        for j, up in enumerate(self.ups):
            skip = skips[-(j + 1)]
            h = self._modulate(self.depth + 1 + j, up(h, skip), metadata)
        return self.head(h)


# Backward compatibility for early notebooks and checkpoints that imported the
# architecture as ``UNet``. New code should use the project name above.
UNet = Waves2SurfNet
