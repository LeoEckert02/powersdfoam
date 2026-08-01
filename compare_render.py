"""Rasterizer vs. ray-tracer comparison strip with render timings + geometry.

For each test frame this renders a single labeled row:

    rasterize RGB | raytrace RGB | 5x|diff| | normal | depth

with the measured per-backend render time drawn on the two RGB tiles and
cross-PSNR(raster, raytrace) on the diff tile. All frames are stacked into one
PNG under ``<checkpoint>/compare/``.

Notes:
  * Both RGB timings use the rgb-only ``benchmark`` path (apples to apples, and
    the same CUDA-event, warmup+reps method as benchmark.py). The view-dependent
    colour query is inside the timed region, matching benchmark.py.
  * Normal and depth come from the rasterizer's ``visualize`` (the ray-tracer
    kernel only writes colour), so those two buffers are rasterized regardless.
  * Backend setup mirrors check_consistency.py / benchmark.py exactly:
    rasterization uses the alpha-complex adjacency; ray tracing adds Steiner
    points and uses the full adjacency.

Usage (on the GPU machine):
    python compare_render.py -c output/<run>/config.yaml --frames 3 --reps 20
"""

import os

import configargparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm, font_manager

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
def prepare_backend(model, args, render_type):
    """Static (view-independent) render inputs for one backend.

    Mirrors check_consistency.render_views: rasterization uses the alpha-complex
    adjacency; ray tracing uses the full adjacency and zeroes Steiner-point
    density (invisible scaffolding). Everything is detached — Warp kernels
    reject tensors that require grad.
    """
    points = model.points.detach()
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

    return dict(
        points=points, radii=radii, density=density, normals=normals,
        texel_sites=texel_sites, texel_height=texel_height,
        att_sites=att_sites, att_values=att_values, att_temps=att_temps,
        adjacency=adjacency, adjacency_offsets=adjacency_offsets,
        adjacency_diff=adjacency_diff,
    )


@torch.no_grad()
def render_rgb(model, args, prep, camera, render_type):
    """One rgb-only render through the given backend (view-dependent)."""
    texel_rgb = model.sv.forward(
        prep["texel_sites"].view(-1, 3).detach(),
        camera,
        prep["att_sites"], prep["att_values"], prep["att_temps"],
    ).view(model.points.shape[0], args.num_texel_sites, 3)

    if render_type == "rasterize":
        rgb = model.rasterizer.benchmark(
            camera, prep["points"], prep["radii"], prep["density"],
            prep["normals"], prep["texel_sites"], texel_rgb,
            prep["texel_height"], prep["adjacency"],
            prep["adjacency_offsets"], prep["adjacency_diff"], 1e-2,
        )
    else:
        eye = camera.eye.to(model.device)
        dists = torch.linalg.norm(prep["points"] - eye[None, :], dim=-1)
        start = int(torch.argmin(dists**2 - prep["radii"] ** 2))
        rgb = model.raytracer.benchmark(
            camera, start, prep["points"], prep["radii"], prep["density"],
            prep["normals"], prep["texel_sites"], texel_rgb,
            prep["texel_height"], prep["adjacency"],
            prep["adjacency_offsets"], prep["adjacency_diff"], 1e-2,
        )
    return rgb.float().clamp(0.0, 1.0)


@torch.no_grad()
def time_render(model, args, prep, camera, render_type, reps, warmup=3):
    """(rgb, milliseconds-per-render) via CUDA events, warmup + reps."""
    for _ in range(warmup):
        rgb = render_rgb(model, args, prep, camera, render_type)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        rgb = render_rgb(model, args, prep, camera, render_type)
    end.record()
    torch.cuda.synchronize()
    return rgb, start.elapsed_time(end) / reps


@torch.no_grad()
def raster_normal_depth(model, prep, camera):
    """Rasterized (normal, depth) buffers via rasterizer.visualize."""
    texel_rgb = model.sv.forward(
        prep["texel_sites"].view(-1, 3).detach(),
        camera,
        prep["att_sites"], prep["att_values"], prep["att_temps"],
    ).view(model.points.shape[0], model.args.num_texel_sites, 3)
    color, depth, normal, alpha, _ = model.rasterizer.visualize(
        camera, prep["points"], prep["radii"], prep["density"], prep["normals"],
        prep["texel_sites"], texel_rgb, prep["texel_height"],
        prep["adjacency"], prep["adjacency_offsets"],
    )
    return normal, depth


# ----------------------------- image composition -----------------------------

def _font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def to_uint8(t):
    return (t.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


def normal_to_uint8(normal):
    return to_uint8((normal + 1) * 0.5)


def depth_to_uint8(depth):
    d = depth.detach().cpu()[:, :, 0]
    valid = d[d > 0]
    dmin = valid.min() if valid.numel() else torch.tensor(0.0)
    dmax = d.max()
    d = (d - dmin) / (dmax - dmin + 1e-8)
    return (cm.viridis(d.numpy())[:, :, :3] * 255).astype(np.uint8)


def label_tile(arr, text):
    """RGB uint8 array -> PIL image with a text label in a dark box, top-left."""
    im = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(max(12, im.width // 28))
    lines = text.split("\n")
    pad = 4
    lh = font.getbbox("Ag")[3] + 2
    box_w = max(draw.textlength(l, font=font) for l in lines) + 2 * pad
    box_h = lh * len(lines) + pad
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    for i, l in enumerate(lines):
        draw.text((pad, pad + i * lh), l, fill=(255, 255, 0), font=font)
    return im


def main():
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, help="Config path")
    parser.add_argument("--frames", type=int, default=3, help="Test frames to show")
    parser.add_argument("--reps", type=int, default=20, help="Timing repetitions")
    args = parser.parse_args()
    params = get_params(args)

    wp.init()
    checkpoint = args.config.replace("/config.yaml", "")
    out_dir = os.path.join(checkpoint, "compare")
    os.makedirs(out_dir, exist_ok=True)

    dh = DataHandler(params)
    dh.reload("test", downsample=params.downsample[-1])
    cameras = dh.cameras

    model = PowerSDFoamScene(params, attr_dtype="half")
    model.initialize_from_dataset(dh, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    model.declare_optimizers(params, params.iterations)
    model.sort_points()

    n = len(cameras)
    step = max(1, n // args.frames)
    frame_ids = list(range(0, n, step))[: args.frames]

    # --- rasterize first (before Steiner points are added) ------------------
    print(f"Rasterizing {len(frame_ids)} frames ({args.reps} reps each)...")
    prep_r = prepare_backend(model, params, "rasterize")
    raster = {}
    for i in frame_ids:
        rgb, ms = time_render(model, params, prep_r, cameras[i], "rasterize", args.reps)
        normal, depth = raster_normal_depth(model, prep_r, cameras[i])
        raster[i] = dict(rgb=rgb, ms=ms, normal=normal, depth=depth)
        print(f"  frame {i:03d}: rasterize {ms:6.2f} ms")

    # --- ray trace (needs Steiner points + full adjacency) ------------------
    print("Adding Steiner points and ray tracing...")
    with torch.no_grad():
        sp, sr = get_steiner_points(
            model.points, model.get_radii().to(torch.float32), cameras
        )
        add_steiner_points(model, sp, sr, model.tscalar)
        model.sort_points()
    prep_t = prepare_backend(model, params, "raytrace")
    raytrace = {}
    for i in frame_ids:
        rgb, ms = time_render(model, params, prep_t, cameras[i], "raytrace", args.reps)
        raytrace[i] = dict(rgb=rgb, ms=ms)
        print(f"  frame {i:03d}: raytrace  {ms:6.2f} ms")

    # --- compose one labeled strip per frame, stacked vertically ------------
    rows = []
    for i in frame_ids:
        r, t = raster[i], raytrace[i]
        cross = psnr(r["rgb"], t["rgb"]).item()
        speedup = r["ms"] / t["ms"] if t["ms"] > 0 else float("nan")
        diff = (5 * (r["rgb"] - t["rgb"]).abs())

        tiles = [
            label_tile(to_uint8(r["rgb"]), f"rasterize\n{r['ms']:.1f} ms"),
            label_tile(to_uint8(t["rgb"]), f"raytrace\n{t['ms']:.1f} ms  ({speedup:.2f}x)"),
            label_tile(to_uint8(diff), f"diff x5\nPSNR {cross:.1f} dB"),
            label_tile(normal_to_uint8(r["normal"]), "normal"),
            label_tile(depth_to_uint8(r["depth"]), "depth"),
        ]
        w = sum(im.width for im in tiles)
        h = max(im.height for im in tiles)
        row = Image.new("RGB", (w, h), (20, 20, 20))
        x = 0
        for im in tiles:
            row.paste(im, (x, 0))
            x += im.width
        rows.append(row)
        print(
            f"frame {i:03d}: raster {r['ms']:.2f} ms | raytrace {t['ms']:.2f} ms "
            f"| {speedup:.2f}x | cross-PSNR {cross:.2f} dB"
        )

    W = max(r.width for r in rows)
    H = sum(r.height for r in rows)
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    out_path = os.path.join(out_dir, "compare.png")
    sheet.save(out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
