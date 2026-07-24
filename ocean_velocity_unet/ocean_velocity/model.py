"""A compact 2-D U-Net with interchangeable metadata conditioning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _normalization(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown normalization {kind!r}")


class DoubleConv(nn.Module):
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
        return self.layers(x)


class Up(nn.Module):
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
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class FeatureModulation(nn.Module):
    """Residual FiLM: ``x * (1 + gamma(z)) + beta(z)``."""

    def __init__(self, metadata_width: int, channels: int) -> None:
        super().__init__()
        self.generator = nn.Linear(metadata_width, 2 * channels)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.generator(metadata).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class MetadataMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.layers(metadata)


class UNet(nn.Module):
    """Four-level U-Net mapping ``[B, C_in, H, W]`` to ``[B, C_out, H, W]``.

    ``base_channels=8`` is the recommended lightweight default. Group
    normalization is robust to the small per-device batches common in
    high-resolution geophysical training.
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
    ) -> None:
        super().__init__()
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

        widths = [base_channels * 2**i for i in range(4)]
        self.stem = DoubleConv(
            stem_channels, widths[0], normalization=normalization
        )
        self.down1 = DoubleConv(
            widths[0], widths[1], stride=2, normalization=normalization
        )
        self.down2 = DoubleConv(
            widths[1], widths[2], stride=2, normalization=normalization
        )
        self.down3 = DoubleConv(
            widths[2], widths[3], stride=2, normalization=normalization
        )
        self.bottleneck = DoubleConv(
            widths[3], widths[3] * 2, stride=2, normalization=normalization
        )
        self.up3 = Up(widths[3] * 2, widths[3], widths[3], normalization)
        self.up2 = Up(widths[3], widths[2], widths[2], normalization)
        self.up1 = Up(widths[2], widths[1], widths[1], normalization)
        self.up0 = Up(widths[1], widths[0], widths[0], normalization)
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        if conditioning == "film":
            film_widths = [*widths, widths[3] * 2, *widths[::-1]]
            self.film = nn.ModuleList(
                FeatureModulation(metadata_width, channels)
                for channels in film_widths
            )
        else:
            self.film = None

    @staticmethod
    def _broadcast(metadata: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return metadata[:, :, None, None].expand(
            -1, -1, x.shape[-2], x.shape[-1]
        )

    def _modulate(
        self, index: int, x: torch.Tensor, metadata: torch.Tensor | None
    ) -> torch.Tensor:
        if self.film is None:
            return x
        return self.film[index](x, metadata)

    def forward(
        self, x: torch.Tensor, metadata: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.conditioning in {"extra_channels", "broadcast"}:
            if metadata is None:
                raise ValueError("Metadata tensor is required by the configured model")
            encoded = self.metadata_encoder(metadata)
            x = torch.cat((x, self._broadcast(encoded, x)), dim=1)
        elif self.conditioning == "film":
            if metadata is None:
                raise ValueError("Metadata tensor is required by the configured model")
            metadata = self.metadata_encoder(metadata)

        x0 = self.stem(x)
        x0 = self._modulate(0, x0, metadata)
        x1 = self.down1(x0)
        x1 = self._modulate(1, x1, metadata)
        x2 = self.down2(x1)
        x2 = self._modulate(2, x2, metadata)
        x3 = self.down3(x2)
        x3 = self._modulate(3, x3, metadata)
        x4 = self.bottleneck(x3)
        x4 = self._modulate(4, x4, metadata)
        y3 = self._modulate(5, self.up3(x4, x3), metadata)
        y2 = self._modulate(6, self.up2(y3, x2), metadata)
        y1 = self._modulate(7, self.up1(y2, x1), metadata)
        y0 = self._modulate(8, self.up0(y1, x0), metadata)
        return self.head(y0)
