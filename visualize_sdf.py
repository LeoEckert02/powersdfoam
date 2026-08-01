"""Visualize the learned SDF of a PowerSDFoam checkpoint.

When an extracted mesh looks wrong, this answers whether the *SDF itself* is
sane or whether the mesh extraction is the problem. It writes, under
``<checkpoint>/sdf_viz/``:

  * three axis-aligned 2D slices of the SDF through the box center, as
    diverging heatmaps (blue = outside / positive, red = inside / negative)
    with the zero level set drawn as a solid black contour and the +/- band
    edges as dashed contours;
  * a histogram of SDF values at the power sites;
  * printed statistics (min/max/mean, % negative, zero-crossing check, NaNs)
    at both the power sites and each slice.

A healthy SDF: a clear spread of values crossing zero, a *tight* band around
a zero contour that traces recognizable geometry, and negative values inside
objects. Symptoms of a broken/undertrained SDF: everything one sign (no
interior), a featureless blob/sphere zero contour, a very wide fuzzy band, or
NaNs.

All outputs are PNG/text -- viewable on the Mac in Preview, no 3D viewer or
local GPU needed.

Usage (on the GPU machine):
    python visualize_sdf.py -c output/<run>/config.yaml
    python visualize_sdf.py -c output/<run>/config.yaml --resolution 512 \
        --bbox_scale 0.3
    python visualize_sdf.py -c output/<run>/config.yaml \
        --bbox_min="-3.16,0.37,1.84" --bbox_max="1.84,5.37,6.84"

The --bbox_* / --bbox_scale / --bbox_q flags behave exactly as in
extract_mesh.py -- crop the slices to the same box you extract a mesh over so
the two line up. --band sets the dashed contour levels (default matches a
scene-scale-appropriate surface band; widen it for large-coordinate scenes).
"""

import os

import configargparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from extract_mesh import load_model, resolve_bounds, report_bounds


@torch.no_grad()
def sdf_slice(model, lo, hi, axis, coord, res):
    """SDF on an axis-aligned plane (fixed `axis` = `coord`), shape (res,res).

    Returns the value grid plus the two varying axis indices (row, col).
    """
    dims = [d for d in (0, 1, 2) if d != axis]
    a = torch.linspace(float(lo[dims[0]]), float(hi[dims[0]]), res, device=model.device)
    b = torch.linspace(float(lo[dims[1]]), float(hi[dims[1]]), res, device=model.device)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    pts = torch.empty(res * res, 3, device=model.device)
    pts[:, dims[0]] = aa.reshape(-1)
    pts[:, dims[1]] = bb.reshape(-1)
    pts[:, axis] = coord
    vals = model.get_sdf(pts).view(res, res).float().cpu().numpy()
    return vals, dims


def sdf_stats(name, vals):
    finite = vals[np.isfinite(vals)]
    n_nan = vals.size - finite.size
    if finite.size == 0:
        print(f"[{name}] ALL non-finite ({n_nan} NaN/inf) -- SDF is broken")
        return
    neg = float((finite < 0).mean())
    crossing = finite.min() < 0 < finite.max()
    print(
        f"[{name}] min={finite.min():.4f} max={finite.max():.4f} "
        f"mean={finite.mean():.4f} | {neg*100:.1f}% negative | "
        f"zero-crossing={'yes' if crossing else 'NO (suspicious)'}"
        + (f" | {n_nan} NaN/inf!" if n_nan else "")
    )


def main():
    from configs import add_group, Params

    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, help="Config path")
    parser.add_argument("--resolution", type=int, default=512, help="Slice resolution")
    parser.add_argument("--band", type=float, default=0.05,
                        help="Dashed contour band edges at +/- this SDF value "
                             "(scale up for large-coordinate scenes)")
    # Same bounds/cropping controls as extract_mesh.py.
    parser.add_argument("--bbox_min", type=str, default=None,
                        help="Slice box min as 'x,y,z' (with --bbox_max)")
    parser.add_argument("--bbox_max", type=str, default=None,
                        help="Slice box max as 'x,y,z' (with --bbox_min)")
    parser.add_argument("--bbox_scale", type=float, default=1.0,
                        help="Scale the box about its center (e.g. 0.3)")
    parser.add_argument("--bbox_q", type=float, default=0.005,
                        help="Quantile for automatic bounds")
    args = parser.parse_args()
    params = get_params(args)

    model, checkpoint = load_model(params, args.config)
    out_dir = os.path.join(checkpoint, "sdf_viz")
    os.makedirs(out_dir, exist_ok=True)

    report_bounds(model, args)
    lo, hi = resolve_bounds(model, args)
    center = 0.5 * (lo + hi)

    # --- stats at the power sites (cheap, direct sanity check) --------------
    with torch.no_grad():
        site_sdf = model.get_sdf().float().cpu().numpy()
    sdf_stats("sites", site_sdf)

    # --- three slices through the box center --------------------------------
    axis_names = ["x", "y", "z"]
    for axis in (0, 1, 2):
        coord = float(center[axis])
        vals, dims = sdf_slice(model, lo, hi, axis, coord, args.resolution)
        sdf_stats(f"slice {axis_names[axis]}={coord:.2f}", vals)

        # Symmetric diverging scale around 0, clipped to the bulk range so the
        # near-surface structure stays visible even if far corners are huge.
        finite = vals[np.isfinite(vals)]
        vmax = float(np.percentile(np.abs(finite), 95)) if finite.size else 1.0
        vmax = max(vmax, 1e-6)

        # imshow extent maps columns->dims[1], rows->dims[0].
        extent = [float(lo[dims[1]]), float(hi[dims[1]]),
                  float(lo[dims[0]]), float(hi[dims[0]])]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(vals, origin="lower", extent=extent, cmap="RdBu",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.contour(vals, levels=[0.0], colors="black", linewidths=1.2,
                   extent=extent, origin="lower")
        ax.contour(vals, levels=[-args.band, args.band], colors="black",
                   linewidths=0.4, linestyles="dashed", extent=extent,
                   origin="lower")
        ax.set_title(f"SDF slice {axis_names[axis]}={coord:.2f}  "
                     f"(solid=zero level set, dashed=+/-{args.band})")
        ax.set_xlabel(axis_names[dims[1]])
        ax.set_ylabel(axis_names[dims[0]])
        fig.colorbar(im, ax=ax, label="signed distance")
        path = os.path.join(out_dir, f"slice_{axis_names[axis]}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # --- histogram of SDF at the sites --------------------------------------
    finite = site_sdf[np.isfinite(site_sdf)]
    if finite.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(finite, bins=120, color="#4477aa")
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title("SDF value distribution at power sites")
        ax.set_xlabel("signed distance")
        ax.set_ylabel("count")
        path = os.path.join(out_dir, "hist_sites.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    print(f"\nSDF visualizations written to {out_dir}/")


if __name__ == "__main__":
    main()
