import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import coords_grid


class GlobalMatchingInit(nn.Module):
    """
    All-pairs global matching at 1/8 resolution for optical flow initialization.

    Pipeline:
      1. Project + downsample features from (in_dim, H/2, W/2) → (dim, H/8, W/8)
         via Conv2d with stride=4 (4× spatial reduction in one step)
      2. Single-layer cross-attention: frame1 queries attend to frame2 keys/values
      3. Build (N, H8*W8, H8*W8) cost volume via cosine similarity
      4. Soft-argmax over target positions
      5. Scale by gamma (init=0) → no-op at start, safe for finetuning from checkpoint

    Args:
        in_dim:          input feature channels — iter_dim of BuildMatchFlow (features at H/2)
        dim:             internal projection dimension (128)
        nheads:          attention heads (4)
        scaling_factor: spatial upsampling applied to the output flow (default 4: H/8 → H/2)
    """

    def __init__(self, in_dim, dim=128, nheads=4, scaling_factor=4):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.proj = nn.Conv2d(in_dim, dim, kernel_size=scaling_factor, stride=scaling_factor, padding=0, bias=False)
        self.cross_attn = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        """
        f1, f2: (N, in_dim, H/2, W/2) — encoder features at 1/2 resolution
        Returns: flow (N, 2, H/8, W/8) in 1/8-scale pixel units
        """
        # --- feature projection + downsample (stride=4: H/2 → H/8) ---
        f1 = self.proj(f1)   # (N, dim, H8, W8)
        f2 = self.proj(f2)
        N, _, H, W = f1.shape

        # --- cross-attention enrichment (frame1 queries, frame2 keys/values) ---
        f1_flat = f1.flatten(2).permute(0, 2, 1)   # (N, H8*W8, dim)
        f2_flat = f2.flatten(2).permute(0, 2, 1)

        f1_attn, _ = self.cross_attn(f1_flat, f2_flat, f2_flat)
        f1_enh = self.norm(f1_flat + f1_attn)       # residual (f1_flat) + layernorm (f1_attn)

        # back to (N, dim, H8*W8)
        f1_enh = f1_enh.permute(0, 2, 1)
        f2_flat = f2_flat.permute(0, 2, 1)

        # --- cosine similarity cost volume (N, H8*W8, H8*W8) ---
        f1_n = F.normalize(f1_enh, dim=1)
        f2_n = F.normalize(f2_flat, dim=1)
        cost = torch.bmm(f1_n.permute(0, 2, 1), f2_n)   # (N, src, tgt)

        # --- soft-argmax: weighted sum of target pixel coordinates ---
        cost_soft = F.softmax(cost, dim=-1)               # (N, H8*W8, H8*W8)

        grid = coords_grid(N, H, W, device=f1.device)    # (N, 2, H8, W8), (x,y) order
        grid_flat = grid.flatten(2).permute(0, 2, 1)     # (N, H8*W8, 2)

        matched = torch.bmm(cost_soft, grid_flat)         # (N, H8*W8, 2)
        matched = matched.permute(0, 2, 1).reshape(N, 2, H, W)

        flow = self.gamma * (matched - grid)              # displacement at 1/8 scale

        return self.scaling_factor * F.interpolate(
            flow, scale_factor=self.scaling_factor, mode='bilinear', align_corners=True
        )
