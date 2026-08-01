"""SDF-based mesh extraction for PowerSDFoam checkpoints.

Two extraction methods that use the learned SDF (requires a model trained
with ``use_sdf: true``):

  * ``mc``    — marching cubes on the neural SDF over the scene bounds.
                Produces a watertight, high-resolution surface.
  * ``cells`` — SDFoam-style explicit extraction: power cells whose sites
                lie in a band around the SDF zero level set are selected
                and their cell polyhedra are emitted directly, preserving
                the trained foam topology.

For the TSDF-fusion baseline (which does not need the SDF), use
``render.py`` as in the original PowerFoam codebase.

Usage:
    python extract_mesh.py -c output/<run>/config.yaml --method mc
    python extract_mesh.py -c output/<run>/config.yaml --method cells \
        --sdf_min -0.02 --sdf_max 0.05
"""

import os

import configargparse
import numpy as np
import torch
from scipy.spatial import KDTree
from skimage import measure

from powersdfoam.power_diagram import extract_cell_meshes


def load_model(args, config_path):
    # Imported lazily so the extraction functions stay usable without the
    # GPU rendering stack.
    import warp as wp
    from data_loader import DataHandler
    from powersdfoam.scene import PowerSDFoamScene

    wp.init()
    checkpoint = config_path.replace("/config.yaml", "")

    data_handler = DataHandler(args)
    data_handler.reload("test", downsample=args.downsample[-1])

    model = PowerSDFoamScene(args)
    model.initialize_from_dataset(data_handler, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")

    if not model.use_sdf:
        raise ValueError(
            "extract_mesh.py needs a model trained with use_sdf: true. "
            "For SDF-free checkpoints use the TSDF pipeline in render.py."
        )
    return model, checkpoint


def site_colors(model):
    """Approximate per-cell albedo from the mean spherical-Voronoi color."""
    n = model.points.shape[0]
    rgb = model.texel_sv_rgb.detach().float()
    rgb = rgb.view(n, model.args.num_texel_sites, model.args.sv_dof, 3)
    return (0.5 + rgb.mean(dim=(1, 2))).clamp(0.0, 1.0).cpu().numpy()


def _parse_xyz(s):
    """Parse an 'x,y,z' string into a length-3 float list (or None)."""
    if s is None:
        return None
    parts = [float(v) for v in s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'x,y,z', got {s!r}")
    return parts


def cropping_requested(args):
    return (
        (_parse_xyz(args.bbox_min) is not None
         and _parse_xyz(args.bbox_max) is not None)
        or args.bbox_scale != 1.0
    )


def resolve_bounds(model, args, pad=0.05):
    """Axis-aligned extraction bounds, with optional user cropping.

    Priority: explicit ``--bbox_min``/``--bbox_max`` if both given, else a
    robust quantile box over the sites (``--bbox_q``) with padding. The
    result is then scaled about its center by ``--bbox_scale`` (1.0 = no
    change, < 1 crops tighter around the center — useful for isolating the
    foreground of an unbounded 360 scene from its background shell).
    """
    pts = model.points.detach().float()
    q = args.bbox_q
    auto_lo = torch.quantile(pts, q, dim=0)
    auto_hi = torch.quantile(pts, 1 - q, dim=0)
    span = (auto_hi - auto_lo).norm()
    auto_lo = auto_lo - pad * span
    auto_hi = auto_hi + pad * span

    bbox_min = _parse_xyz(args.bbox_min)
    bbox_max = _parse_xyz(args.bbox_max)
    if bbox_min is not None and bbox_max is not None:
        lo = torch.tensor(bbox_min, dtype=pts.dtype, device=pts.device)
        hi = torch.tensor(bbox_max, dtype=pts.dtype, device=pts.device)
    else:
        lo, hi = auto_lo, auto_hi

    if args.bbox_scale != 1.0:
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * args.bbox_scale
        lo, hi = center - half, center + half

    return lo, hi


def report_bounds(model, args):
    """Print site centroid + full auto bounds as a starting point for crops.

    Called unconditionally by both extractors so you always have reference
    coordinates to craft an explicit --bbox_min/--bbox_max, even on a run
    with no cropping.
    """
    pts = model.points.detach().float()
    q = args.bbox_q
    auto_lo = torch.quantile(pts, q, dim=0)
    auto_hi = torch.quantile(pts, 1 - q, dim=0)
    centroid = pts.mean(dim=0)
    print(f"Site centroid:      {centroid.tolist()}")
    print(f"Full auto bounds:   {auto_lo.tolist()} .. {auto_hi.tolist()}")


@torch.no_grad()
def sdf_on_grid(model, bbox_min, bbox_max, resolution, chunk=2**20):
    xs = torch.linspace(bbox_min[0], bbox_max[0], resolution, device=model.device)
    ys = torch.linspace(bbox_min[1], bbox_max[1], resolution, device=model.device)
    zs = torch.linspace(bbox_min[2], bbox_max[2], resolution, device=model.device)
    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

    vals = torch.empty(pts.shape[0], device=model.device)
    for i in range(0, pts.shape[0], chunk):
        vals[i : i + chunk] = model.get_sdf(pts[i : i + chunk])
    return vals.view(resolution, resolution, resolution).cpu().numpy()


def extract_marching_cubes(model, args):
    report_bounds(model, args)
    bbox_min, bbox_max = resolve_bounds(model, args)
    print(f"Marching cubes over {bbox_min.tolist()} .. {bbox_max.tolist()}")

    volume = sdf_on_grid(model, bbox_min, bbox_max, args.mc_resolution)
    if volume.min() > 0 or volume.max() < 0:
        raise RuntimeError("SDF has no zero crossing inside the scene bounds")

    bbox_min = bbox_min.cpu().numpy()
    bbox_max = bbox_max.cpu().numpy()
    spacing = (bbox_max - bbox_min) / (args.mc_resolution - 1)
    verts, faces, normals, _ = measure.marching_cubes(
        volume, level=0.0, spacing=tuple(spacing)
    )
    verts = verts + bbox_min[None, :]

    # Color vertices from the nearest power cell's mean texture color
    kdtree = KDTree(model.points.detach().float().cpu().numpy())
    _, nn = kdtree.query(verts)
    colors = site_colors(model)[nn]

    return verts, faces, colors, None


def extract_power_cells(model, args):
    report_bounds(model, args)
    with torch.no_grad():
        sdf = model.get_sdf().float().cpu().numpy()
        density = model.get_density().float().cpu().numpy()
        points = model.points.detach().float().cpu().numpy()
        weights = (model.get_radii().detach().float() ** 2).cpu().numpy()

    mask = (sdf > args.sdf_min) & (sdf < args.sdf_max)
    if args.density_min > 0:
        mask &= density > args.density_min
    if cropping_requested(args):
        lo, hi = resolve_bounds(model, args)
        lo, hi = lo.cpu().numpy(), hi.cpu().numpy()
        in_box = np.all((points >= lo) & (points <= hi), axis=1)
        mask &= in_box
        print(f"Crop kept {in_box.sum():,}/{len(points):,} sites in bbox")
    selected = np.where(mask)[0]
    print(f"Selected {len(selected):,}/{len(points):,} cells")
    if len(selected) == 0:
        raise RuntimeError(
            "No cells selected; loosen --sdf_min/--sdf_max/--density_min"
        )

    verts, faces, face_cell = extract_cell_meshes(
        points, weights, selected, pad_ratio=args.pad_ratio
    )

    # Face selection as in SDFoam: keep only triangles whose vertices all
    # lie near the zero level set, dropping interior/exterior cell walls.
    if args.face_sdf_max > 0 and faces.shape[0] > 0:
        with torch.no_grad():
            v = torch.tensor(verts, dtype=torch.float32, device=model.device)
            vert_sdf = model.get_sdf(v).abs().cpu().numpy()
        keep = (vert_sdf[faces] < args.face_sdf_max).all(axis=1)
        faces = faces[keep]
        face_cell = face_cell[keep]
        print(f"Face selection kept {keep.sum():,}/{keep.shape[0]:,} triangles")

    face_colors = site_colors(model)[face_cell] if faces.shape[0] else None
    return verts, faces, None, face_colors


def save_mesh(path, verts, faces, vertex_colors=None, face_colors=None):
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if vertex_colors is not None:
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    if face_colors is not None:
        mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)
    mesh.remove_unreferenced_vertices()
    mesh.export(path)
    print(f"Mesh saved to {path} ({len(mesh.vertices):,} verts, "
          f"{len(mesh.faces):,} faces)")


def main():
    from configs import add_group, Params

    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)

    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )
    parser.add_argument(
        "--method", type=str, default="mc", choices=["mc", "cells"],
        help="mc: marching cubes on the SDF; cells: power-cell extraction",
    )
    parser.add_argument(
        "--mesh_name", type=str, default=None, help="Output mesh filename"
    )
    # Marching-cubes options
    parser.add_argument(
        "--mc_resolution", type=int, default=512, help="MC grid resolution"
    )
    # Extraction bounds / cropping (both methods). Use these to isolate the
    # foreground of an unbounded 360 scene from its background shell.
    parser.add_argument(
        "--bbox_min", type=str, default=None,
        help="Crop box min as 'x,y,z' (use with --bbox_max). Overrides the "
             "automatic quantile bounds. Run once without it to print the "
             "site centroid + full bounds as a starting point.",
    )
    parser.add_argument(
        "--bbox_max", type=str, default=None,
        help="Crop box max as 'x,y,z' (use with --bbox_min).",
    )
    parser.add_argument(
        "--bbox_scale", type=float, default=1.0,
        help="Scale the auto/explicit box about its center (1.0 = no change, "
             "e.g. 0.3 crops to the central 30%% to isolate the foreground).",
    )
    parser.add_argument(
        "--bbox_q", type=float, default=0.005,
        help="Quantile for the automatic axis-aligned bounds (larger drops "
             "more outlier background sites).",
    )
    # Cell-extraction options (SDF band and filters)
    parser.add_argument("--sdf_min", type=float, default=-0.02)
    parser.add_argument("--sdf_max", type=float, default=0.05)
    parser.add_argument(
        "--density_min", type=float, default=0.0,
        help="Optional minimum cell density filter (0 disables)",
    )
    parser.add_argument(
        "--face_sdf_max", type=float, default=0.0,
        help="Keep only triangles with |SDF| below this at all vertices "
             "(0 disables face selection, emitting full cell polyhedra)",
    )
    parser.add_argument(
        "--pad_ratio", type=float, default=0.25,
        help="Bounding padding for the power diagram",
    )

    args = parser.parse_args()
    params = get_params(args)

    model, checkpoint = load_model(params, args.config)

    if args.method == "mc":
        verts, faces, vertex_colors, face_colors = extract_marching_cubes(
            model, args
        )
    else:
        verts, faces, vertex_colors, face_colors = extract_power_cells(
            model, args
        )

    mesh_name = args.mesh_name or f"mesh_{args.method}"
    out_path = os.path.join(checkpoint, f"{mesh_name}.ply")
    save_mesh(out_path, verts, faces, vertex_colors, face_colors)


if __name__ == "__main__":
    main()
