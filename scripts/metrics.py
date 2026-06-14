"""
Stage 3: design-quality and performance metrics for 2D unit-cell images.

Purpose in the project: this is the geometry-to-property evaluator. It labels
training cells and scores generated cells with the SAME homogenization, so a
generated design is judged on performance error against its target property
(not pixel error). Use property_vector() to build training labels and the
conditioning target; use evaluate_against_target() to score a generated cell.

Reports:
  - volume fraction
  - connectivity (components, percolation, periodic connectivity)
  - symmetry (mirror axes + a 0..1 score)
  - manufacturability (thin members, thin gaps, floating bits, trapped voids)
  - effective properties via periodic homogenization: stiffness tensor CH,
    Young's moduli, Poisson's ratio, bulk modulus, shear modulus, and
    stiffness-to-weight (specific stiffness)

Convention: a pixel is SOLID when its value is above 127 (matches the Stage 2
PNGs where solid = 255). Stiffness is RELATIVE to the solid material (solid
E = 1); multiply by your real Young's modulus for physical values.

Usage:
  python metrics.py cell.png                       # readable report
  python metrics.py cell.png --json                # raw dict (incl. CH tensor)
  python metrics.py images/ --out metrics.csv      # whole folder -> CSV
  python metrics.py cell.png --no-fem              # skip homogenization (fast)
  python metrics.py cell.png --min-feature 4       # min printable width (px)
  python metrics.py cell.png --target target.json  # performance error vs target
"""

import os
import csv
import json
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import (label, binary_opening, distance_transform_edt,
                           generate_binary_structure)
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

EIGHT = generate_binary_structure(2, 2)   # 8-connectivity


# ----------------------------- io -----------------------------
def load_binary(path, threshold=127):
    """Read an image and return a 2D uint8 array, 1 = solid, 0 = void."""
    img = np.array(Image.open(path).convert("L"))
    return (img > threshold).astype(np.uint8)


# ----------------------------- basic metrics -----------------------------
def volume_fraction(solid):
    return float(solid.mean())


def symmetry(solid, tol=0.02):
    """Mirror-symmetry flags and a single 0..1 score (1 = perfectly symmetric)."""
    def match(a, b):
        return 1.0 - float(np.mean(a != b))
    sx = match(solid, np.fliplr(solid))   # left-right
    sy = match(solid, np.flipud(solid))   # up-down
    sd = match(solid, solid.T) if solid.shape[0] == solid.shape[1] else 0.0
    flags = {"mirror_x": sx >= 1 - tol,
             "mirror_y": sy >= 1 - tol,
             "diagonal": sd >= 1 - tol}
    if flags["mirror_x"] and flags["mirror_y"] and flags["diagonal"]:
        label_ = "square"
    elif flags["mirror_x"] and flags["mirror_y"]:
        label_ = "biaxial"
    elif flags["mirror_x"] or flags["mirror_y"]:
        label_ = "uniaxial"
    elif flags["diagonal"]:
        label_ = "diagonal"
    else:
        label_ = "none"
    return {"symmetry_class": label_,
            "symmetry_score": round(max(sx, sy, sd), 4),
            **flags}


def connectivity(solid):
    """Components, spanning percolation, and periodic (tiling) connectivity."""
    H, W = solid.shape
    lbl, ncomp = label(solid, structure=EIGHT)
    if ncomp == 0:
        return {"n_solid_components": 0, "largest_component_fraction": 0.0,
                "percolates_x": False, "percolates_y": False,
                "periodic_connected_x": False, "periodic_connected_y": False}

    sizes = np.bincount(lbl.ravel())[1:]
    largest_frac = float(sizes.max() / solid.sum())

    perc_x = perc_y = False
    for k in range(1, ncomp + 1):
        comp = lbl == k
        if comp[:, 0].any() and comp[:, -1].any():
            perc_x = True
        if comp[0, :].any() and comp[-1, :].any():
            perc_y = True

    # periodic connectivity: tile 3x3, see if the centre block's solid links
    # to its neighbours across the seam (i.e. the material tiles continuously).
    tiled = np.tile(solid, (3, 3))
    tl, tn = label(tiled, structure=EIGHT)
    centre = tl[H:2 * H, W:2 * W]
    centre_labels = set(np.unique(centre)) - {0}
    left = set(np.unique(tl[H:2 * H, 0:W])) - {0}
    right = set(np.unique(tl[H:2 * H, 2 * W:3 * W])) - {0}
    top = set(np.unique(tl[0:H, W:2 * W])) - {0}
    bottom = set(np.unique(tl[2 * H:3 * H, W:2 * W])) - {0}
    per_x = bool(centre_labels & left & right)
    per_y = bool(centre_labels & top & bottom)

    return {"n_solid_components": int(ncomp),
            "largest_component_fraction": round(largest_frac, 4),
            "percolates_x": bool(perc_x), "percolates_y": bool(perc_y),
            "periodic_connected_x": per_x, "periodic_connected_y": per_y}


# ----------------------------- manufacturability -----------------------------
def _disk(radius):
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def manufacturability(solid, min_feature_px=3):
    """Flag thin members, thin gaps, floating material, and trapped voids."""
    void = 1 - solid
    total = solid.size
    solid_n = max(int(solid.sum()), 1)
    void_n = max(int(void.sum()), 1)
    r = max(1, min_feature_px // 2)
    se = _disk(r)

    # thin solid members: removed by an opening that keeps features >= 2r wide
    opened = binary_opening(solid, structure=se)
    thin_solid_frac = float((solid & ~opened).sum() / solid_n)

    # thin gaps between members: same test on the void phase
    opened_v = binary_opening(void, structure=se)
    thin_void_frac = float((void & ~opened_v).sum() / void_n)

    # floating material: solid not part of the largest connected component
    lbl, ncomp = label(solid, structure=EIGHT)
    if ncomp > 1:
        sizes = np.bincount(lbl.ravel())[1:]
        biggest = int(sizes.argmax()) + 1
        floating_frac = float((solid & (lbl != biggest)).sum() / solid_n)
    else:
        floating_frac = 0.0

    # trapped voids: void regions that do not touch the image border
    vlbl, vn = label(void, structure=EIGHT)
    border = set(np.unique(np.concatenate([
        vlbl[0, :], vlbl[-1, :], vlbl[:, 0], vlbl[:, -1]]))) - {0}
    trapped = np.isin(vlbl, list(set(range(1, vn + 1)) - border)) & (vlbl > 0)
    trapped_void_frac = float(trapped.sum() / total)

    # estimated minimum solid thickness: 2 * smallest ridge of the distance map
    dt = distance_transform_edt(solid)
    ridge = dt[(dt > 0) & (dt >= 0.5)]
    min_thickness = float(2 * ridge.min()) if ridge.size else 0.0

    ok = (thin_solid_frac < 0.02 and floating_frac < 0.01
          and trapped_void_frac < 0.02)
    return {"min_feature_px": min_feature_px,
            "thin_solid_fraction": round(thin_solid_frac, 4),
            "thin_void_fraction": round(thin_void_frac, 4),
            "floating_solid_fraction": round(floating_frac, 4),
            "trapped_void_fraction": round(trapped_void_frac, 4),
            "est_min_solid_thickness_px": round(min_thickness, 2),
            "manufacturable": bool(ok)}


# ----------------------------- homogenization -----------------------------
def _element_stiffness(nu0=0.3):
    """Q4 plane-stress element stiffness for a unit square, E0 = 1."""
    a = b = 0.5
    C0 = (1.0 / (1 - nu0 ** 2)) * np.array(
        [[1, nu0, 0], [nu0, 1, 0], [0, 0, (1 - nu0) / 2.0]])
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    ke = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dNxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dNeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            dNx, dNy = dNxi / a, dNeta / b
            B = np.zeros((3, 8))
            B[0, 0::2] = dNx
            B[1, 1::2] = dNy
            B[2, 0::2] = dNy
            B[2, 1::2] = dNx
            ke += (B.T @ C0 @ B) * (a * b)   # weight 1 per Gauss point
    # affine nodal fields that produce the three unit macro strains
    chi0 = np.array([
        [-a, 0, a, 0, a, 0, -a, 0],       # exx = 1  -> u_x = x
        [0, -b, 0, -b, 0, b, 0, b],       # eyy = 1  -> u_y = y
        [-b, 0, -b, 0, b, 0, b, 0],       # gxy = 1  -> u_x = y
    ]).T                                   # shape (8, 3)
    return ke, chi0


def homogenize(solid, nu0=0.3, void_scale=1e-9):
    """Return the effective plane-stress stiffness tensor CH (E0 = 1) and
    derived effective Young's moduli, Poisson's ratio, and shear modulus."""
    nely, nelx = solid.shape
    nel = nelx * nely
    ke, chi0 = _element_stiffness(nu0)

    # element scale: 1 for solid, tiny for void (keeps the system non-singular)
    scale = np.where(solid.ravel(order="C") > 0, 1.0, void_scale)
    # build elements in row-major (ey outer, ex inner) to match solid.ravel
    ex = np.tile(np.arange(nelx), nely)
    ey = np.repeat(np.arange(nely), nelx)

    def nid(ix, iy):
        return iy * (nelx + 1) + ix
    bl = nid(ex, ey); br = nid(ex + 1, ey)
    tr = nid(ex + 1, ey + 1); tl = nid(ex, ey + 1)
    nodes = np.stack([bl, br, tr, tl], axis=1)          # (nel, 4)
    edof = np.empty((nel, 8), dtype=np.int64)
    edof[:, 0::2] = 2 * nodes
    edof[:, 1::2] = 2 * nodes + 1

    # periodic dof map: collapse right edge onto left, top onto bottom
    nnodes = (nelx + 1) * (nely + 1)
    iy_all, ix_all = np.divmod(np.arange(nnodes), nelx + 1)
    pid = (iy_all % nely) * nelx + (ix_all % nelx)
    dofmap = np.empty(2 * nnodes, dtype=np.int64)
    dofmap[0::2] = 2 * pid
    dofmap[1::2] = 2 * pid + 1
    edofp = dofmap[edof]
    ndof = 2 * nelx * nely

    # assemble K
    keflat = ke.ravel()
    sK = (keflat[None, :] * scale[:, None]).ravel()
    iK = np.repeat(edofp, 8, axis=1).ravel()
    jK = np.tile(edofp, (1, 8)).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

    # assemble F (3 macro-strain load cases): fe = scale * ke @ chi0
    fe = ke @ chi0                                   # (8, 3)
    F = np.zeros((ndof, 3))
    contrib = scale[:, None, None] * fe[None, :, :]  # (nel, 8, 3)
    for c in range(3):
        np.add.at(F[:, c], edofp.ravel(), contrib[:, :, c].ravel())

    # solve with the first node fixed to remove rigid-body translation
    free = np.arange(2, ndof)
    chi = np.zeros((ndof, 3))
    chi[free, :] = spsolve(K[free][:, free], F[free, :])

    # effective tensor: CH_ij = (1/V) sum_e scale_e (chi0 - chi_e)_i ke (..)_j
    chi_e = chi[edofp]                               # (nel, 8, 3)
    chi_e_t = np.transpose(chi_e, (0, 2, 1))         # (nel, 3, 8)
    d = chi0.T[None, :, :] - chi_e_t                 # (nel, 3, 8)
    V = float(nelx * nely)
    CH = np.zeros((3, 3))
    keK = ke
    for i in range(3):
        di = d[:, i, :]                              # (nel, 8)
        dik = di @ keK                               # (nel, 8)
        for j in range(3):
            dj = d[:, j, :]
            CH[i, j] = np.sum(scale * np.einsum("ek,ek->e", dik, dj)) / V

    CH = 0.5 * (CH + CH.T)                            # symmetrize tiny asymmetry
    S = np.linalg.inv(CH) if np.linalg.cond(CH) < 1e12 else None
    if S is not None:
        Ex = 1.0 / S[0, 0]
        Ey = 1.0 / S[1, 1]
        nu_xy = -S[0, 1] / S[0, 0]
    else:
        Ex = Ey = nu_xy = 0.0
    # 2D (area) bulk modulus: response to equibiaxial strain
    K2d = (CH[0, 0] + CH[1, 1] + 2 * CH[0, 1]) / 4.0
    return {"CH": [[round(v, 6) for v in row] for row in CH.tolist()],
            "E_eff_x_rel": round(float(Ex), 5),
            "E_eff_y_rel": round(float(Ey), 5),
            "poisson_eff": round(float(nu_xy), 4),
            "G_eff_rel": round(float(CH[2, 2]), 5),
            "K_eff_rel": round(float(K2d), 5),
            "auxetic": bool(nu_xy < -0.01)}


# ----------------------------- top level -----------------------------
def compute_metrics(solid, do_fem=True, min_feature_px=3, nu0=0.3):
    out = {"volume_fraction": round(volume_fraction(solid), 4)}
    out.update(symmetry(solid))
    out.update(connectivity(solid))
    out.update(manufacturability(solid, min_feature_px))
    if do_fem:
        if solid.sum() == 0:
            out.update({"E_eff_x_rel": 0.0, "E_eff_y_rel": 0.0,
                        "poisson_eff": 0.0, "G_eff_rel": 0.0,
                        "K_eff_rel": 0.0, "auxetic": False})
        else:
            try:
                out.update(homogenize(solid, nu0=nu0))
            except Exception as e:
                out.update({"fem_error": str(e)})
        # stiffness-to-weight (Slide 5 target): effective modulus per unit density
        vf = max(out["volume_fraction"], 1e-6)
        e_mean = 0.5 * (out.get("E_eff_x_rel", 0.0) + out.get("E_eff_y_rel", 0.0))
        out["specific_stiffness_rel"] = round(e_mean / vf, 5)
    return out


# canonical property vector: the SAME quantities that label the training data
# and condition the generator, so achieved-vs-target error is well defined.
PROPERTY_KEYS = ["volume_fraction", "E_eff_x_rel", "E_eff_y_rel",
                 "poisson_eff", "K_eff_rel", "G_eff_rel"]


def property_vector(solid, nu0=0.3):
    """Return only the conditioning/target properties for a cell."""
    m = compute_metrics(solid, do_fem=True, nu0=nu0)
    return {k: m[k] for k in PROPERTY_KEYS if k in m}


def evaluate_against_target(metrics, target):
    """Compare achieved properties to a target dict (Slide 8: performance error).
    Returns per-property absolute and relative error plus a mean relative error.
    Only keys present in `target` are scored."""
    errs = {}
    rels = []
    for k, tv in target.items():
        if k not in metrics:
            continue
        av = float(metrics[k])
        abs_e = abs(av - float(tv))
        # normalize by the larger magnitude so targets near 0 (e.g. Poisson)
        # do not blow up the relative error
        denom = max(abs(float(tv)), abs(av), 1e-6)
        rel_e = abs_e / denom
        errs[k] = {"target": float(tv), "achieved": round(av, 5),
                   "abs_error": round(abs_e, 5), "rel_error": round(rel_e, 4)}
        rels.append(rel_e)
    errs["mean_rel_error"] = round(float(np.mean(rels)), 4) if rels else None
    return errs


def report(path, m):
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"volume fraction        {m['volume_fraction']}")
    print(f"symmetry               {m['symmetry_class']} "
          f"(score {m['symmetry_score']})")
    print(f"solid components       {m['n_solid_components']} "
          f"(largest {m['largest_component_fraction']})")
    print(f"percolates x / y       {m['percolates_x']} / {m['percolates_y']}")
    print(f"periodic-connected x/y {m['periodic_connected_x']} / "
          f"{m['periodic_connected_y']}")
    print(f"thin solid fraction    {m['thin_solid_fraction']}")
    print(f"floating solid frac.   {m['floating_solid_fraction']}")
    print(f"trapped void frac.     {m['trapped_void_fraction']}")
    print(f"min solid thickness    ~{m['est_min_solid_thickness_px']} px")
    print(f"manufacturable         {m['manufacturable']}")
    if "E_eff_x_rel" in m and "poisson_eff" in m:
        print(f"effective E (x / y)    {m['E_eff_x_rel']} / {m['E_eff_y_rel']} "
              f"(relative to solid)")
        print(f"effective Poisson      {m['poisson_eff']}  "
              f"auxetic={m['auxetic']}")
        print(f"effective bulk K       {m.get('K_eff_rel')} (relative)")
        print(f"effective shear G      {m['G_eff_rel']} (relative)")
        print(f"specific stiffness     {m.get('specific_stiffness_rel')} "
              f"(E/density, stiffness-to-weight)")


def main():
    ap = argparse.ArgumentParser(description="Unit-cell design-quality metrics")
    ap.add_argument("path", help="image file or folder of images")
    ap.add_argument("--no-fem", action="store_true", help="skip homogenization")
    ap.add_argument("--min-feature", type=int, default=3,
                    help="minimum printable feature width in pixels")
    ap.add_argument("--nu0", type=float, default=0.3,
                    help="Poisson's ratio of the solid material")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--out", default=None, help="CSV output path (folder mode)")
    ap.add_argument("--target", default=None,
                    help="JSON file or string of target properties; reports "
                         "achieved-vs-target performance error (single image)")
    args = ap.parse_args()
    do_fem = not args.no_fem

    if os.path.isdir(args.path):
        files = sorted(f for f in os.listdir(args.path)
                       if f.lower().endswith((".png", ".bmp", ".tif", ".jpg")))
        rows = []
        for f in files:
            m = compute_metrics(load_binary(os.path.join(args.path, f)),
                                do_fem, args.min_feature, args.nu0)
            m = {"filename": f, **m}
            m.pop("CH", None)            # keep the CSV flat
            rows.append(m)
        out = args.out or os.path.join(args.path, "metrics.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote metrics for {len(rows)} images to {out}")
    else:
        m = compute_metrics(load_binary(args.path), do_fem,
                            args.min_feature, args.nu0)
        if args.target:
            if os.path.isfile(args.target):
                target = json.load(open(args.target))
            else:
                target = json.loads(args.target)
            ev = evaluate_against_target(m, target)
            if args.json:
                print(json.dumps({"metrics": m, "evaluation": ev}, indent=2))
            else:
                report(args.path, m)
                print("\n--- performance error vs target ---")
                for k, v in ev.items():
                    if k == "mean_rel_error":
                        print(f"mean relative error    {v}")
                    else:
                        print(f"{k:22s} target {v['target']}  achieved "
                              f"{v['achieved']}  rel.err {v['rel_error']}")
        elif args.json:
            print(json.dumps(m, indent=2))
        else:
            report(args.path, m)


if __name__ == "__main__":
    main()
