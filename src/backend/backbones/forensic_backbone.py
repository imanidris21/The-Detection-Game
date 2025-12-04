"""
forensic_backbone.py - Enhanced forensic backbone with SRM and DCT analysis
Combines multiple low-level forensic features for AI image detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from .base_backbone import BaseBackbone


class SRMLayer(nn.Module):
    """
    Spatial Rich Model (SRM) layer with pre-defined high-pass filters
    Contains 32 different SRM kernels for detecting various artifacts
    This class has been written with the assistance of Claude code AI. All suggestions were reviewed critically and modified as needed.
    """

    def __init__(self):
        super().__init__()

        # Define SRM kernels (30 filters from SRM literature)
        srm_kernels = self._get_srm_kernels()

        # Convert to torch tensors and register as buffers (non-trainable)
        self.num_filters = len(srm_kernels)
        for i, kernel in enumerate(srm_kernels):
            self.register_buffer(f'srm_kernel_{i}', torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0))

        # Learnable weights for combining SRM responses
        self.srm_weights = nn.Parameter(torch.ones(self.num_filters) / self.num_filters)

    def _get_srm_kernels(self):
        """Get the 32 SRM filter kernels"""
        kernels = []

        # 1st order SRM kernels
        kernels.extend([
            # Horizontal/Vertical 1st order
            [[-1, 2, -1]],
            [[-1], [2], [-1]],

            # Diagonal 1st order
            [[-1, 0, 1], [0, 0, 0], [1, 0, -1]],
            [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],

            # 2nd order horizontal/vertical
            [[1, -2, 1]],
            [[1], [-2], [1]],

            # Edge detection kernels
            [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
            [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],

            # Square 3x3 kernels
            [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],

            # 5x5 kernels (simplified)
            [
                [0, 0, 1, 0, 0],
                [0, 1, -4, 1, 0],
                [1, -4, 6, -4, 1],
                [0, 1, -4, 1, 0],
                [0, 0, 1, 0, 0]
            ],

            # Additional edge and texture detection kernels
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  # Sobel X
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],  # Sobel Y
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],  # Laplacian
            [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],  # Custom edge

            # Texture detection
            [[1, -1, 1], [-1, 1, -1], [1, -1, 1]],
            [[-1, 1, -1], [1, -1, 1], [-1, 1, -1]],

            # More sophisticated patterns
            [[0, 1, 2, 1, 0], [1, 0, -2, 0, 1], [2, -2, -8, -2, 2], [1, 0, -2, 0, 1], [0, 1, 2, 1, 0]],

            # Additional high-pass filters
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],

            # Noise detection patterns
            [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],

            # Compression artifact detection
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],

            # Final specialized kernels
            [[1, -2, 1, -2, 1]],
            [[1], [-2], [1], [-2], [1]],
            [[2, -1, 0, -1, 2]],
            [[2], [-1], [0], [-1], [2]]
        ])

        return kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SRM filters to input image
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            SRM response maps (B, num_filters, H, W)
        """
        batch_size, channels, height, width = x.shape

        # Convert to grayscale if needed
        if channels == 3:
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            x_gray = x

        responses = []
        target_h, target_w = height, width  # Target output size

        for i in range(self.num_filters):
            kernel = getattr(self, f'srm_kernel_{i}')
            # Apply convolution with padding to maintain size
            # Handle non-square kernels by calculating padding for each dimension
            kernel_h = kernel.shape[-2]  # height
            kernel_w = kernel.shape[-1]  # width

            # Calculate padding to get output size >= target size
            # For odd kernels: pad_total = kernel_size - 1
            # For even kernels: pad_total = kernel_size
            pad_h_total = kernel_h - 1
            pad_w_total = kernel_w - 1

            # Distribute padding (for even kernels, add extra to right/bottom)
            pad_h_top = pad_h_total // 2
            pad_h_bottom = pad_h_total - pad_h_top
            pad_w_left = pad_w_total // 2
            pad_w_right = pad_w_total - pad_w_left

            padding = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)  # (left, right, top, bottom)

            # Pad input and apply convolution
            x_padded = F.pad(x_gray, padding, mode='reflect')
            response = F.conv2d(x_padded, kernel, padding=0)

            # Crop to exact target size if needed (handles off-by-one from even kernels)
            if response.shape[-2] != target_h or response.shape[-1] != target_w:
                response = response[:, :, :target_h, :target_w]

            responses.append(response)

        # Stack all responses - all should be (B, 1, H, W) now
        srm_responses = torch.cat(responses, dim=1)  # (B, num_filters, H, W)

        return srm_responses


class DCTBlock(nn.Module):
    """
    DCT block processing for detecting compression artifacts
    Extracts DCT coefficients from 8x8 blocks similar to JPEG compression
    This class has been written with the assistance of Claude code AI. All suggestions were reviewed critically and modified as needed.
    """

    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size

        # Pre-compute DCT basis functions
        self.register_buffer('dct_basis', self._get_dct_basis(block_size))

    def _get_dct_basis(self, N: int) -> torch.Tensor:
        """Compute 2D DCT basis functions"""
        basis = torch.zeros(N, N, N, N)
        for u in range(N):
            for v in range(N):
                for x in range(N):
                    for y in range(N):
                        cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
                        cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
                        basis[u, v, x, y] = (2.0 / N) * cu * cv * \
                                          np.cos((2 * x + 1) * u * np.pi / (2 * N)) * \
                                          np.cos((2 * y + 1) * v * np.pi / (2 * N))
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract DCT coefficients from 8x8 blocks (VECTORIZED - GPU optimized)
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            DCT coefficients (B, C*64, H//8, W//8)
        
        """
        batch_size, channels, height, width = x.shape
        block_size = self.block_size

        # Ensure dimensions are divisible by block_size
        h_blocks = height // block_size
        w_blocks = width // block_size

        if h_blocks == 0 or w_blocks == 0:
            # If image is too small, use direct DCT
            return self._direct_dct(x)

        # Crop to fit blocks exactly
        x_cropped = x[:, :, :h_blocks * block_size, :w_blocks * block_size]

        # Reshape into blocks: (B, C, h_blocks, block_size, w_blocks, block_size)
        x_blocks = x_cropped.view(
            batch_size, channels, h_blocks, block_size, w_blocks, block_size
        )
        # Rearrange to: (B, C, h_blocks, w_blocks, block_size, block_size)
        x_blocks = x_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()

        # Reshape for batch matrix multiplication
        # (B, C, h_blocks, w_blocks, block_size, block_size) -> (B*C*h_blocks*w_blocks, block_size, block_size)
        num_blocks = batch_size * channels * h_blocks * w_blocks
        x_blocks_flat = x_blocks.view(num_blocks, block_size, block_size)

        # VECTORIZED DCT using matrix multiplication
        # DCT(X) = sum over x,y of X[x,y] * basis[u,v,x,y]
        # This can be computed as: basis[u,v] @ X where basis is reshaped properly
        # For 2D DCT: Y = T @ X @ T^T where T is the DCT transform matrix

        # Create DCT transform matrix from basis (only need to do this once per dimension)
        # T[u, x] = basis[u, 0, x, 0] for the 1D case
        # For 2D: we use the full basis reshaped

        # Reshape basis for einsum: (block_size, block_size, block_size, block_size)
        # We want: dct_block[u,v] = sum_{x,y} block[x,y] * basis[u,v,x,y]
        # This is equivalent to: einsum('...xy,uvxy->...uv', block, basis)

        dct_basis_device = self.dct_basis.to(x.device)

        # Use einsum for vectorized DCT computation
        # x_blocks_flat: (num_blocks, block_size, block_size)
        # dct_basis: (block_size, block_size, block_size, block_size)
        # Result: (num_blocks, block_size, block_size)
        dct_coeffs_flat = torch.einsum('nxy,uvxy->nuv', x_blocks_flat, dct_basis_device)

        # Reshape back to original block structure
        dct_coeffs = dct_coeffs_flat.view(batch_size, channels, h_blocks, w_blocks, block_size, block_size)

        # Reshape for output: (B, C*64, h_blocks, w_blocks)
        dct_output = dct_coeffs.view(
            batch_size, channels * block_size * block_size, h_blocks, w_blocks
        )

        return dct_output

    def _direct_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT directly to small images"""
        # For very small images, just return flattened version
        return x.view(x.size(0), -1, 1, 1)


class ForensicBackbone(BaseBackbone):
    """
    Forensic backbone combining SRM filters and DCT analysis for AI image detection
    """

    def __init__(
        self,
        device: Optional[str] = None,
        feature_dim: int = 768,  # Match DINOv2
        use_srm: bool = True,
        use_dct: bool = True,
        conv_layers: int = 2,
        conv_channels: Tuple[int, ...] = (128, 256)
    ):
        """
        Args:
            device: Device to run on
            feature_dim: Output feature dimension
            use_srm: Include SRM filter responses
            use_dct: Include DCT block analysis
            conv_layers: Number of learnable conv layers
            conv_channels: Conv layer channel dimensions
        """
        super().__init__(device)

        self.feature_dim = feature_dim
        self.use_srm = use_srm
        self.use_dct = use_dct
        self.architecture_family = 'forensic'
        self.normalization_type = 'imagenet'
        self.input_size = (224, 224)

        # Initialize forensic components
        self.srm_layer = SRMLayer() if use_srm else None
        self.dct_block = DCTBlock() if use_dct else None

        # Calculate input channels for conv layers
        conv_input_channels = 0
        if use_srm:
            conv_input_channels += 32  # 32 SRM filters
        if use_dct:
            conv_input_channels += 192  # 3 RGB channels × 64 DCT coefficients per 8x8 block

        # Learnable convolutional layers
        conv_layers_list = []
        in_channels = conv_input_channels

        for i, out_channels in enumerate(conv_channels):
            conv_layers_list.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2) if i < len(conv_channels) - 1 else nn.AdaptiveAvgPool2d((1, 1))
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Final projection to target feature dimension
        final_conv_dim = conv_channels[-1] if conv_channels else conv_input_channels
        self.feature_projection = nn.Linear(final_conv_dim, feature_dim)

        # Move to device
        self.to(self.device)

        print(f"Initialized ForensicBackbone:")
        print(f"  SRM: {use_srm}, DCT: {use_dct}")
        print(f"  Conv layers: {conv_layers}, channels: {conv_channels}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Device: {device}")

    def _load_model(self):
        """Load model (not needed for forensic backbone)"""
        pass

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract forensic features from images"""
        feature_maps = []
        target_device = images.device

        # SRM features
        if self.use_srm:
            srm_features = self.srm_layer(images)  # (B, 32, H, W)
            srm_features = srm_features.to(target_device)
            feature_maps.append(srm_features)

        # DCT features
        if self.use_dct:
            dct_features = self.dct_block(images)  # (B, 192, H//8, W//8)
            dct_features = dct_features.to(target_device)
            # Upsample to match other feature map sizes
            dct_features = F.interpolate(
                dct_features, size=images.shape[2:], mode='bilinear', align_corners=False
            )
            feature_maps.append(dct_features)

        # Concatenate all feature maps
        if feature_maps:
            combined_features = torch.cat(feature_maps, dim=1)
            combined_features = combined_features.to(target_device)
        else:
            # Fallback: use input images
            combined_features = images

        # Apply learnable conv layers
        processed_features = self.conv_layers(combined_features)  # (B, final_dim, 1, 1)

        # Flatten and project
        flattened = processed_features.view(processed_features.size(0), -1)
        final_features = self.feature_projection(flattened)

        # Ensure final features are on correct device
        final_features = final_features.to(target_device)
        return final_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass (supports both frozen and trainable modes)"""
        images = images.to(self.device)

        # For neural mode, don't use no_grad
        if self.training:
            features = self._extract_features(images)
        else:
            with torch.no_grad():
                features = self._extract_features(images)

        # L2 normalize
        features = F.normalize(features, dim=1)

        return features.detach().cpu() if not self.training else features

    def get_info(self) -> dict:
        """Get forensic backbone information"""
        info = super().get_info()
        info.update({
            'use_srm': self.use_srm,
            'use_dct': self.use_dct,
            'supports_text': False,
            'supports_patches': False,
            'srm_filters': 30 if self.use_srm else 0,
            'dct_block_size': 8 if self.use_dct else 0,
        })
        return info


# Factory function
def create_forensic_backbone(
    feature_dim: int = 768,
    device: Optional[str] = None,
    **kwargs
) -> ForensicBackbone:
    """Create forensic backbone with default parameters"""
    return ForensicBackbone(
        device=device,
        feature_dim=feature_dim,
        **kwargs
    )