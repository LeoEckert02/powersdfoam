# PowerSDFoam

A merge of two methods:

- **[PowerFoam](https://github.com/theialab/powerfoam)** — bounded power diagrams with a rasterizer and a ray tracer.
- **[SDFoam](https://github.com/mmlab-cv/SDFoam)** — a neural SDF attached to the foam.

Each cell's density is derived from the SDF instead of being a free parameter, so
gradients from both renderers train the SDF. Setting `use_sdf: false` in a config
recovers plain PowerFoam.

## Install

Needs Linux, an NVIDIA GPU (Compute Capability ≥ 7.0) and CUDA 12.x.

```bash
conda create -n powersdfoam python=3.11 -y
conda activate powersdfoam

pip install torch==2.9.1 torchvision==0.24.1
pip install -r requirements.txt
```

## Train

```bash
python train.py -c configs/dtu.yaml
```

Configs live in `configs/`. Results are written to `output/<scene>@<hash>/`,
containing `model.pt`, `config.yaml` and test renders.

## Evaluate

```bash
python test.py -c output/<run>/config.yaml        # PSNR / SSIM / LPIPS
python benchmark.py -c output/<run>/config.yaml   # rendering speed
python view.py -c output/<run>/config.yaml        # interactive viewer
```

## Extract a mesh

Needs a model trained with `use_sdf: true`.

```bash
# marching cubes on the SDF — smooth, watertight
python extract_mesh.py -c output/<run>/config.yaml --method mc --mc_resolution 512

# power cells near the zero level set — the explicit foam surface
python extract_mesh.py -c output/<run>/config.yaml --method cells \
    --sdf_min -0.1 --sdf_max 0.1 --face_sdf_max 0.01
```

Both accept `--bbox_min "x,y,z" --bbox_max "x,y,z"` or `--bbox_scale 0.3` to crop
to a region (useful for unbounded scenes with a large background shell).

## Inspect the SDF

```bash
python visualize_sdf.py -c output/<run>/config.yaml
```

Writes 2D slices of the SDF and a histogram to `output/<run>/sdf_viz/`, and prints
value statistics. Use this when a mesh looks wrong to check whether the SDF itself
is healthy: values should span a real range, cross zero, and the zero contour
should trace the object.

## Compare the two renderers

```bash
python check_consistency.py -c output/<run>/config.yaml --frames 5
```

Renders the same views with the rasterizer and the ray tracer and reports PSNR
between them. They should agree (> ~35 dB) — this tests renderer equivalence,
not reconstruction quality.

```bash
python compare_render.py -c output/<run>/config.yaml --frames 3
```

One image per frame: rasterized | ray traced | difference | normals | depth,
with render times drawn on it.

## Credits

Built on [PowerFoam](https://github.com/theialab/powerfoam) (Govindarajan et al.)
and [SDFoam](https://github.com/mmlab-cv/SDFoam) (Rech et al.).
