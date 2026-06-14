"""
Stage 2: Dataset preparation for 2D unit-cell images.

Generates binary 64x64 unit cells across six pattern families and writes a
labels.csv with: volume fraction, connectivity, symmetry, approximate
stiffness class, and an auxetic-like flag.

Convention: pixel value 1 = solid material, 0 = void.
PNGs are saved as 0 (black void) / 255 (white solid).

The labels for volume fraction, connectivity, and symmetry are MEASURED from
the image. The stiffness class is a HEURISTIC proxy (volume fraction +
percolation), not a finite-element result. The auxetic-like flag is set BY
CONSTRUCTION from the generator family, since re-entrant geometry is what makes
a cell auxetic-like. Replace these two with FEA homogenization later in Stage 4.
"""

import os
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_dilation

# ----------------------------- configuration -----------------------------
SIZE = 64                 # image side length in pixels
N_PER_CATEGORY = 80       # samples per pattern family
SEED = 42
OUT_DIR = "/home/claude/unitcells"
IMG_DIR = os.path.join(OUT_DIR, "images")
CSV_PATH = os.path.join(OUT_DIR, "labels.csv")
SYM_TOL = 0.02            # allowed mismatch fraction for symmetry detection
MIN_VF, MAX_VF = 0.12, 0.88   # reject degenerate (near-empty / near-full) cells

rng = np.random.default_rng(SEED)

# ----------------------------- helpers -----------------------------
YY, XX = np.mgrid[0:SIZE, 0:SIZE]


def quadrant_symmetrize(a):
    """Force biaxial (x and y mirror) symmetry by folding the top-left quadrant."""
    h = SIZE // 2
    q = a[:h, :h]
    top = np.hstack([q, np.fliplr(q)])
    return np.vstack([top, np.flipud(top)]).astype(np.uint8)


def draw_strut(a, p0, p1, thickness):
    """Set solid for all pixels within thickness/2 of segment p0->p1."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy + 1e-9
    t = ((XX - x0) * dx + (YY - y0) * dy) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    dist2 = (XX - px) ** 2 + (YY - py) ** 2
    a[dist2 <= (thickness / 2.0) ** 2] = 1


# ----------------------------- pattern families -----------------------------
def gen_holes(rng):
    """Solid plate with circular holes punched out."""
    a = np.ones((SIZE, SIZE), dtype=np.uint8)
    n = rng.integers(2, 6)
    for _ in range(n):
        cx, cy = rng.integers(8, SIZE - 8, size=2)
        r = rng.integers(6, 16)
        a[(XX - cx) ** 2 + (YY - cy) ** 2 <= r * r] = 0
    return a


def gen_lattice(rng):
    """Grid of horizontal and vertical bars enclosing void cells."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    period = int(rng.integers(16, 28))
    thick = int(rng.integers(4, 9))
    offset = int(rng.integers(0, period))
    a[((YY + offset) % period) < thick] = 1
    a[((XX + offset) % period) < thick] = 1
    # keep an outer frame so the cell tiles cleanly
    a[:thick, :] = 1; a[-thick:, :] = 1
    a[:, :thick] = 1; a[:, -thick:] = 1
    return a


def gen_cross(rng):
    """Central plus / cross shape."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    c = SIZE // 2
    half = int(rng.integers(4, 9))
    arm = int(rng.integers(20, 31))
    a[c - half:c + half, c - arm:c + arm] = 1
    a[c - arm:c + arm, c - half:c + half] = 1
    return a


def gen_ring(rng):
    """Annulus with four struts tying it to the cell edges."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    c = SIZE / 2.0
    r_out = rng.integers(18, 27)
    r_in = r_out - rng.integers(5, 10)
    r2 = (XX - c) ** 2 + (YY - c) ** 2
    a[(r2 <= r_out * r_out) & (r2 >= r_in * r_in)] = 1
    thick = int(rng.integers(4, 8))
    # tie struts to edges (vertical and horizontal connectors)
    a[int(c - thick / 2):int(c + thick / 2), :] = 1
    a[:, int(c - thick / 2):int(c + thick / 2)] = 1
    # re-punch the central void
    a[r2 < r_in * r_in] = 0
    return a


def gen_bars(rng):
    """Random connected struts between boundary anchor points."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    thick = float(rng.integers(4, 8))
    anchors = [(0, SIZE // 2), (SIZE - 1, SIZE // 2),
               (SIZE // 2, 0), (SIZE // 2, SIZE - 1),
               (SIZE // 2, SIZE // 2)]
    n = rng.integers(3, 6)
    for _ in range(n):
        p0 = anchors[rng.integers(0, len(anchors))]
        p1 = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        draw_strut(a, p0, p1, thick)
    return a


def gen_reentrant(rng):
    """Re-entrant (auxetic-like) cell built from inward-angled struts."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    thick = float(rng.integers(4, 7))
    c = SIZE // 2
    depth = int(rng.integers(8, 16))     # how far the vertices point inward
    # build the top-left quadrant of a re-entrant (bowtie) cell, then mirror
    draw_strut(a, (0, 0), (c - depth, c), thick)        # corner to inner vertex
    draw_strut(a, (c - depth, c), (c, c), thick)        # inner vertex to centre
    draw_strut(a, (c, 0), (c - depth, c), thick)        # top edge to inner vertex
    a = quadrant_symmetrize(a)
    return a


GENERATORS = {
    "holes": gen_holes,
    "lattice": gen_lattice,
    "cross": gen_cross,
    "ring": gen_ring,
    "bars": gen_bars,
    "reentrant": gen_reentrant,
}
AUXETIC_FAMILIES = {"reentrant"}


# ----------------------------- label computation -----------------------------
def volume_fraction(a):
    return float(a.mean())


def symmetry_label(a):
    sx = np.mean(a != np.fliplr(a)) <= SYM_TOL   # left-right mirror
    sy = np.mean(a != np.flipud(a)) <= SYM_TOL   # up-down mirror
    sd = np.mean(a != a.T) <= SYM_TOL            # main-diagonal mirror
    if sx and sy and sd:
        return "square"
    if sx and sy:
        return "biaxial"
    if sx or sy:
        return "uniaxial"
    if sd:
        return "diagonal"
    return "none"


def connectivity_info(a):
    struct = np.ones((3, 3), dtype=int)          # 8-connectivity
    lbl, ncomp = label(a, structure=struct)
    perc_x = perc_y = False
    for k in range(1, ncomp + 1):
        comp = lbl == k
        cols = np.where(comp.any(axis=0))[0]
        rows = np.where(comp.any(axis=1))[0]
        if cols.size and cols.min() == 0 and cols.max() == SIZE - 1:
            perc_x = True
        if rows.size and rows.min() == 0 and rows.max() == SIZE - 1:
            perc_y = True
    if ncomp == 0:
        cclass = "empty"
    elif ncomp == 1:
        cclass = "single"
    elif ncomp <= 3:
        cclass = "few"
    else:
        cclass = "many"
    return ncomp, cclass, perc_x, perc_y


def stiffness_class(vf, perc_x, perc_y):
    """Rough proxy. Load-bearing needs a percolating solid path."""
    if not (perc_x or perc_y):
        return "very_low"
    if not (perc_x and perc_y):
        return "low"
    if vf < 0.35:
        return "low"
    if vf < 0.55:
        return "medium"
    return "high"


# ----------------------------- main loop -----------------------------
def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rows = []
    idx = 0
    for family, gen in GENERATORS.items():
        made = 0
        attempts = 0
        while made < N_PER_CATEGORY and attempts < N_PER_CATEGORY * 20:
            attempts += 1
            a = gen(rng)
            # randomly enforce biaxial symmetry on half the non-reentrant cells
            if family != "reentrant" and rng.random() < 0.5:
                a = quadrant_symmetrize(a)
            a = (a > 0).astype(np.uint8)
            vf = volume_fraction(a)
            if vf < MIN_VF or vf > MAX_VF:
                continue
            ncomp, cclass, px, py = connectivity_info(a)
            fname = f"{family}_{made:03d}.png"
            Image.fromarray((a * 255).astype(np.uint8), mode="L").save(
                os.path.join(IMG_DIR, fname))
            rows.append({
                "filename": fname,
                "family": family,
                "volume_fraction": round(vf, 4),
                "n_components": ncomp,
                "connectivity": cclass,
                "percolates_x": int(px),
                "percolates_y": int(py),
                "symmetry": symmetry_label(a),
                "stiffness_class": stiffness_class(vf, px, py),
                "auxetic_like": int(family in AUXETIC_FAMILIES),
            })
            made += 1
            idx += 1

    fields = ["filename", "family", "volume_fraction", "n_components",
              "connectivity", "percolates_x", "percolates_y", "symmetry",
              "stiffness_class", "auxetic_like"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} images to {IMG_DIR}")
    print(f"labels at {CSV_PATH}")
    return rows


if __name__ == "__main__":
    main()
