"""
Spinodoid unit-cell dataset generator (Option C: clean, curved, connected data).

The v2 model looks "random and blocky" mainly because its training data is
hand-drawn clip-art (straight struts + circles) from generate_unitcells.py: a
diffusion model can only reproduce and blend its data distribution. This script
replaces that distribution with the SPINODOID family of Kumar, Tan, Zheng &
Kochmann, "Inverse-designed spinodoid metamaterials," npj Comp. Mater. (2020):
each cell is a level set of an anisotropic Gaussian random field (a sum of
cosine waves whose wave-vectors are restricted to direction cones). The result
is smooth, curved, bicontinuous and ALWAYS well connected, and the cone angle
tunes anisotropy -> a natural spread of stiffness. It is the smooth relative of
the gaussianField() code already in the procedural demo.

Output is drop-in for ddpm2.py: images/<name>.png (0=void, 255=solid) plus a
labels_cond.csv with columns filename, volume_fraction, E_mean_rel. Labels are
MEASURED with metrics.homogenize (same evaluator ddpm2/metrics use elsewhere).

Usage:
  python generate_spinodoid.py --n 6000 --size 32 --out /home/claude/unitcells_spino
  python generate_spinodoid.py --n 24 --montage montage.png   # quick visual QA
"""
import os, csv, argparse
import numpy as np
from PIL import Image

# reuse the project's homogenizer so labels match the rest of the pipeline
import metrics

# direction-cone presets (degrees, measured from the +x axis, folded to [0,180)).
# a narrow cone about one axis -> lamellar/columnar (anisotropic, stiff along it);
# both axes -> cubic; a wide cone -> isotropic. Mixing these spans the stiffness
# range the conditioner needs to learn.
CONE_PRESETS = {
    "isotropic": (dict(axes=[0.0, 90.0], half_angle=90.0)),
    "cubic":     (dict(axes=[0.0, 90.0], half_angle=22.0)),
    "columnar_x": (dict(axes=[0.0],        half_angle=18.0)),
    "columnar_y": (dict(axes=[90.0],       half_angle=18.0)),
    "diagonal":  (dict(axes=[45.0, 135.0], half_angle=22.0)),
}
PRESET_NAMES = list(CONE_PRESETS.keys())


def _candidate_waves(size, beta, band, axes, half_angle):
    """Integer wave-vectors (nx, ny) with frequency in [beta-band, beta+band]
    whose direction lies within half_angle of one of the allowed axes. Using
    integer frequencies (cycles per cell) makes every wave exactly periodic, so
    the cell tiles seamlessly."""
    R = int(np.ceil(beta + band)) + 1
    out = []
    for nx in range(-R, R + 1):
        for ny in range(-R, R + 1):
            if nx == 0 and ny == 0:
                continue
            f = np.hypot(nx, ny)
            if f < beta - band or f > beta + band:
                continue
            ang = np.degrees(np.arctan2(ny, nx)) % 180.0
            near = min(abs(((ang - ax + 90.0) % 180.0) - 90.0) for ax in axes)
            if near <= half_angle:
                out.append((nx, ny))
    return np.array(out, dtype=np.float64) if out else np.zeros((0, 2))


def spinodoid_field(size, res, beta, band, cone, n_waves, rng):
    """Anisotropic Gaussian random field = normalized sum of cosine waves,
    evaluated on a res x res grid spanning one [0,size) cell. Frequencies are
    integer cycles-per-cell so the field is exactly periodic; sampling at res >
    size supersamples it, which anti-aliases the level-set boundary."""
    cand = _candidate_waves(size, beta, band, cone["axes"], cone["half_angle"])
    if len(cand) == 0:
        cand = _candidate_waves(size, beta, band, [0.0, 90.0], 90.0)
    k = min(n_waves, len(cand))
    ks = cand[rng.choice(len(cand), size=k, replace=False)]
    ph = rng.uniform(0.0, 2.0 * np.pi, size=k)
    coord = np.arange(res) * (size / res)     # cell coords in [0,size)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    field = np.zeros((res, res), dtype=np.float64)
    for (nx, ny), p in zip(ks, ph):
        field += np.cos(2.0 * np.pi * (nx * xx / size + ny * yy / size) + p)
    return field * np.sqrt(2.0 / max(k, 1))   # unit-variance GRF (Kumar 2020)


def make_cell(size, rng, ss=4, beta_lo=1.4, beta_hi=2.8):
    """One random spinodoid cell + the parameters used to make it. Coarse
    features (low beta) so a 32 px cell reads as smooth curved blobs rather than
    fine noise; supersampled by `ss` then area-downsampled for clean edges.
    Lower beta = larger features (easier for a small model to resolve cleanly)."""
    beta = rng.uniform(beta_lo, beta_hi)      # feature frequency (cycles/cell)
    preset = PRESET_NAMES[rng.integers(0, len(PRESET_NAMES))]
    cone = CONE_PRESETS[preset]
    vf = float(rng.uniform(0.30, 0.62))       # target solid fraction
    res = size * ss
    field = spinodoid_field(size, res, beta, 0.6, cone, n_waves=2000, rng=rng)
    thr = np.quantile(field, vf)              # level set at the target vf
    hi = (field <= thr).astype(np.float64)    # smooth, curved, connected phase
    # area-average back down to `size` and re-threshold -> anti-aliased boundary
    solid = (hi.reshape(size, ss, size, ss).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
    return solid, dict(beta=round(beta, 2), preset=preset)


def main():
    ap = argparse.ArgumentParser(description="Spinodoid unit-cell dataset")
    ap.add_argument("--n", type=int, default=6000, help="number of cells")
    ap.add_argument("--size", type=int, default=32, help="cell resolution (px)")
    ap.add_argument("--out", default="/home/claude/unitcells_spino",
                    help="output dir (images/ + labels_cond.csv)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-lo", type=float, default=1.4,
                    help="min feature frequency (cycles/cell); lower = coarser")
    ap.add_argument("--beta-hi", type=float, default=2.8,
                    help="max feature frequency (cycles/cell)")
    ap.add_argument("--montage", default=None,
                    help="also write an NxN contact sheet here for visual QA")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows, thumbs = [], []
    made = 0
    while made < args.n:
        solid, meta = make_cell(args.size, rng,
                                beta_lo=args.beta_lo, beta_hi=args.beta_hi)
        vf = float(solid.mean())
        if vf < 0.15 or vf > 0.85:
            continue
        # measured connectivity guard: spinodoids are bicontinuous, but the level
        # set can occasionally pinch; keep only cells that percolate both ways
        con = metrics.connectivity(solid)
        if not (con["periodic_connected_x"] and con["periodic_connected_y"]):
            continue
        hom = metrics.homogenize(solid) if solid.sum() > 4 else None
        if hom is None:
            continue
        e_mean = 0.5 * (hom["E_eff_x_rel"] + hom["E_eff_y_rel"])
        fname = f"spino_{made:05d}.png"
        Image.fromarray((solid * 255).astype(np.uint8), mode="L").save(
            os.path.join(img_dir, fname))
        rows.append({"filename": fname,
                     "volume_fraction": round(vf, 4),
                     "E_mean_rel": round(float(e_mean), 6),
                     "preset": meta["preset"], "beta": meta["beta"]})
        if args.montage and len(thumbs) < 64:
            thumbs.append(solid)
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{args.n}")

    with open(os.path.join(args.out, "labels_cond.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "volume_fraction",
                                          "E_mean_rel", "preset", "beta"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} cells to {args.out}")

    if args.montage and thumbs:
        g = int(np.ceil(np.sqrt(len(thumbs))))
        s = args.size
        sheet = np.full((g * (s + 2), g * (s + 2)), 200, np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, g)
            sheet[r * (s + 2):r * (s + 2) + s, c * (s + 2):c * (s + 2) + s] = \
                (1 - t) * 255
        Image.fromarray(sheet, mode="L").resize((g * (s + 2) * 6,) * 2,
            Image.NEAREST).save(args.montage)
        print(f"montage -> {args.montage}")


if __name__ == "__main__":
    main()
