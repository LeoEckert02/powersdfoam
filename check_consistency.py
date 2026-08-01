"""Rasterizer vs. ray-tracer consistency check for PowerSDFoam.

PowerFoam's central invariant is that rasterization and ray tracing of the
same power-diagram scene produce (up to float precision) identical images.
The SDF integration only changes *where per-cell densities come from*, so
the invariant must survive the merge — this script verifies it on a trained
checkpoint by rendering the same test views through both backends and
comparing.

The per-backend scene preparation mirrors benchmark.py exactly:
rasterization uses the alpha-complex adjacency; ray tracing adds Steiner
points and uses the full adjacency (neither affects the rendered integral).

Usage (on the GPU machine):
    python check_consistency.py -c output/<run>/config.yaml --frames 5

Interpretation: PSNR(raster, raytrace) should be well above the PSNR of
either against ground truth (>= ~35 dB; typically much higher).  A low
cross-backend PSNR indicates a real inconsistency in the representation.
"""

import os

import configargparse
import numpy as np
import torch
from PIL import Image

import warp as wp

from configs import *
from data_loader import DataHandler
from powersdfoam.scene import PowerSDFoamScene
from powersdfoam.metrics import psnr
from benchmark import (
    build_power_adjacency,
    get_steiner_points,
    add_steiner_points,
)


@torch.no_grad()
def render_views(model, args, cameras, render_type, frame_ids):
    """Render the given frames with one backend, following benchmark.py."""
    points = model.points
    radii = model.get_radii()

    adjacency, adjacency_offsets = build_power_adjacency(
        points, radii, alpha_complex=(render_type == "rasterize")
    )
    num_adjs = adjacency_offsets.diff()

    pm = 0.5 * (points.norm(dim=-1) ** 2 - radii**2)
    self_points = points.repeat_interleave(num_adjs, dim=0)
    adjacency_diff = points[adjacency] - self_points
    pm_diff = pm[adjacency] - pm.repeat_interleave(num_adjs, dim=0)
    adjacency_diff = torch.cat([adjacency_diff, pm_diff[:, None]], dim=-1)
    adjacency_diff = adjacency_diff.to(torch.float16)

    density = model.get_density()
    if getattr(model, "steiner_mask", None) is not None:
        density = density.clone()
        density[model.steiner_mask] = 0
    normals = model.get_normals()
    tangents, bitangent = model.get_tangents()
    offsets = model.texel_sites * radii[:, None, None]
    offsets = (
        offsets[..., 0:1] * tangents[:, None, :]
        + offsets[..., 1:2] * bitangent[:, None, :]
    )
    texel_sites = model.points[:, None, :] + offsets
    att_sites, att_values, att_temps = model.get_att_sv()
    texel_height = model.texel_height * radii[:, None]

    images = {}
    for i in frame_ids:
        camera = cameras[i]
        texel_rgb = model.sv.forward(
            texel_sites.view(-1, 3).detach(),
            camera,
            att_sites,
            att_values,
            att_temps,
        )
        texel_rgb = texel_rgb.view(
            model.points.shape[0], args.num_texel_sites, 3
        )

        if render_type == "rasterize":
            rgb = model.rasterizer.benchmark(
                camera,
                points,
                radii,
                density,
                normals,
                texel_sites,
                texel_rgb,
                texel_height,
                adjacency,
                adjacency_offsets,
                adjacency_diff,
                1e-2,
            )
        else:
            camera_eye = camera.eye.to(model.device)
            dists = torch.linalg.norm(points - camera_eye[None, :], dim=-1)
            start_point_idx = int(torch.argmin(dists**2 - radii**2))
            rgb = model.raytracer.benchmark(
                camera,
                start_point_idx,
                points,
                radii,
                density,
                normals,
                texel_sites,
                texel_rgb,
                texel_height,
                adjacency,
                adjacency_offsets,
                adjacency_diff,
                1e-2,
            )
        images[i] = rgb.float().clamp(0.0, 1.0)
        torch.cuda.synchronize()

    return images


def main():
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )
    parser.add_argument(
        "--frames", type=int, default=5, help="Number of test frames to compare"
    )
    args = parser.parse_args()
    params = get_params(args)

    wp.init()
    checkpoint = args.config.replace("/config.yaml", "")
    out_dir = os.path.join(checkpoint, "consistency")
    os.makedirs(out_dir, exist_ok=True)

    test_data_handler = DataHandler(params)
    test_data_handler.reload("test", downsample=params.downsample[-1])
    cameras = test_data_handler.cameras

    model = PowerSDFoamScene(params, attr_dtype="half")
    model.initialize_from_dataset(test_data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    model.declare_optimizers(params, params.iterations)
    model.sort_points()

    n = len(cameras)
    step = max(1, n // args.frames)
    frame_ids = list(range(0, n, step))[: args.frames]

    print(f"Rendering {len(frame_ids)} frames with the rasterizer...")
    raster = render_views(model, params, cameras, "rasterize", frame_ids)

    # Steiner points are only used by the ray tracer (benchmark.py order)
    print("Adding Steiner points and rendering with the ray tracer...")
    with torch.no_grad():
        steiner_points, steiner_radii = get_steiner_points(
            model.points, model.get_radii().to(torch.float32), cameras
        )
        add_steiner_points(model, steiner_points, steiner_radii, model.tscalar)
        model.sort_points()
    raytraced = render_views(model, params, cameras, "raytrace", frame_ids)

    print()
    cross_list, rast_list, ray_list = [], [], []
    for i in frame_ids:
        gt = test_data_handler.rgbs[i].cuda().float()
        cross = psnr(raster[i], raytraced[i]).item()
        p_rast = psnr(raster[i], gt).item()
        p_ray = psnr(raytraced[i], gt).item()
        max_diff = (raster[i] - raytraced[i]).abs().max().item()
        cross_list.append(cross)
        rast_list.append(p_rast)
        ray_list.append(p_ray)
        print(
            f"frame {i:03d}: PSNR(raster,raytrace)={cross:6.2f} dB | "
            f"max|diff|={max_diff:.4f} | "
            f"PSNR raster/GT={p_rast:5.2f} raytrace/GT={p_ray:5.2f}"
        )

        panel = torch.cat(
            [raster[i], raytraced[i], 5 * (raster[i] - raytraced[i]).abs()],
            dim=1,
        )
        panel = (panel.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(panel).save(f"{out_dir}/{i:03d}.png")

    mean_cross = sum(cross_list) / len(cross_list)
    print(
        f"\nmean PSNR(raster, raytrace) = {mean_cross:.2f} dB "
        f"(raster/GT {sum(rast_list)/len(rast_list):.2f}, "
        f"raytrace/GT {sum(ray_list)/len(ray_list):.2f})"
    )
    print(f"Side-by-side panels (raster | raytrace | 5x diff): {out_dir}/")
    if mean_cross < 30.0:
        print("WARNING: backends disagree — the representation is inconsistent.")
    else:
        print("OK: both rendering paradigms agree on this checkpoint.")


if __name__ == "__main__":
    main()
