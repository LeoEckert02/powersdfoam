"""Neural signed distance field for PowerSDFoam.

Ported from SDFoam (Rech et al., CVPRW 2026), which in turn adapts the
NeuS SDF network (https://github.com/Totoro97/NeuS).  The network predicts
a global signed distance field f(x) whose zero level set is the scene
surface.  Per-cell volume densities are derived from the SDF via the
NeuS-style logistic density

    rho(x) = s * sigmoid(-s f(x)) * (1 - sigmoid(-s f(x)))

with a single learnable sharpness s (SingleVarianceNetwork).  The density
peaks on the zero level set and decays symmetrically away from it, so the
foam's photometric supervision shapes the SDF while the SDF regularizes
the foam geometry.

The geometric initialization is aligned to the mean of the initialization
point cloud, so the initial sphere is centered on the scene rather than at
the world origin.
"""

import math

import numpy as np
import torch
import torch.nn as nn


def get_embedder(multires: int, input_dims: int = 3):
    """NeRF-style sin/cos positional encoding (raw input is kept)."""
    if multires <= 0:
        return None, input_dims

    def embed(x):
        freq_bands = 2.0 ** torch.linspace(
            0.0, multires - 1, multires, device=x.device, dtype=x.dtype
        )
        outs = [x]
        for f in freq_bands:
            w = 2.0 * math.pi * f
            outs += [torch.sin(x * w), torch.cos(x * w)]
        return torch.cat(outs, dim=-1)

    input_ch = input_dims + 2 * multires * input_dims
    return embed, input_ch


class SDFNetwork(nn.Module):
    """MLP predicting a signed distance value for a 3D point.

    With ``geometric_init`` the network starts out approximating the SDF of
    a sphere of radius ``bias`` centered on ``point_cloud``'s mean (or the
    origin if no point cloud is given).
    """

    def __init__(
        self,
        d_in=3,
        d_out=1,
        d_hidden=256,
        n_layers=8,
        skip_in=(4,),
        multires=6,
        bias=0.5,
        scale=1.0,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
        point_cloud=None,
    ):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        if point_cloud is not None:
            if isinstance(point_cloud, torch.Tensor):
                pc = point_cloud.detach().float().reshape(-1, d_in)
            else:
                pc = torch.tensor(point_cloud, dtype=torch.float32).reshape(-1, d_in)
            self.register_buffer("pc_center", pc.mean(dim=0, keepdim=True).cpu())
        else:
            self.register_buffer("pc_center", torch.zeros(1, d_in))

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        nn.init.constant_(lin.bias, -bias)
                    else:
                        nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif multires > 0 and l in self.skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs - self.pc_center
        inputs = inputs * self.scale
        if self.embed_fn is not None:
            inputs = self.embed_fn(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        # First channel is the SDF; scale back to world units
        return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[..., :1]

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """SDF gradient w.r.t. input positions (keeps the graph for training)."""
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        y = self.sdf(x)
        (grad,) = torch.autograd.grad(
            y, x, torch.ones_like(y), create_graph=True, retain_graph=True
        )
        return grad

    @staticmethod
    def eikonal_loss(grads: torch.Tensor) -> torch.Tensor:
        return ((grads.norm(2, dim=-1) - 1) ** 2).mean()


class SingleVarianceNetwork(nn.Module):
    """Learnable sharpness s = exp(variance) of the SDF-to-density mapping.

    A single scalar shared across the scene, as in NeuS/SDFoam.  Larger s
    concentrates density closer to the zero level set.
    """

    def __init__(self, init_val=0.3):
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(float(init_val)))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.variance).clip(1e-6, 1e6)


def sdf_to_density(sdf: torch.Tensor, inv_s: torch.Tensor, scale: float = 1.0):
    """NeuS-style logistic density: peaks at the zero level set.

    rho = scale * s * sigmoid(-s * f) * (1 - sigmoid(-s * f)), the derivative
    of the sigmoid CDF used by SDFoam's ray tracer.
    """
    sig = torch.sigmoid(-inv_s * sdf)
    return scale * inv_s * sig * (1.0 - sig)
