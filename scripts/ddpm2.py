"""
Diffusion v2: conditional DDPM on the 8040-cell dataset.
Condition = [volume_fraction (1)] + [stiffness one-hot low/med/high (3)] = 4 dims.
Adds EMA of the weights (sampled from the EMA copy) and CFG dropout.

Chunked/resumable trainer keeps each call short.
Usage:
  python ddpm2.py train --epochs 8     # repeat to resume
  python ddpm2.py sample --vf 0.4 --stiff high -n 16 --guidance 2.0
"""
import os, sys, csv, math, copy, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMG = 32
T_STEPS = 400
COND_DIM = 4
STIFF = ["low", "medium", "high"]
DATA = "/home/claude/unitcells_big"
CKPT = "/home/claude/checkpoints/ddpm2.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

betas = torch.linspace(1e-4, 0.02, T_STEPS)
abar = torch.cumprod(1 - betas, dim=0)


def stiff_onehot(e, edges):
    i = 0 if e < edges[0] else (1 if e < edges[1] else 2)
    v = [0., 0., 0.]; v[i] = 1.; return v


class Cells(Dataset):
    def __init__(self):
        rows = list(csv.DictReader(open(os.path.join(DATA, "labels_cond.csv"))))
        E = np.array([float(r["E_mean_rel"]) for r in rows])
        self.edges = (float(np.quantile(E, 1/3)), float(np.quantile(E, 2/3)))
        self.items = []
        for r in rows:
            cond = [float(r["volume_fraction"])] + \
                   stiff_onehot(float(r["E_mean_rel"]), self.edges)
            self.items.append((r["filename"], np.float32(cond)))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        f, cond = self.items[i]
        img = Image.open(os.path.join(DATA, "images", f)).convert("L").resize(
            (IMG, IMG), Image.BILINEAR)
        x = (np.array(img, np.float32) / 255.0 > 0.5).astype(np.float32) * 2 - 1
        return torch.from_numpy(x)[None], torch.from_numpy(cond)


def time_embed(t, dim):
    half = dim // 2
    fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * fr[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class ResBlock(nn.Module):
    def __init__(self, ci, co, ed):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.emb = nn.Linear(ed, co)
        self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.emb(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.sk(x)


class UNet(nn.Module):
    def __init__(self, ed=128):
        super().__init__(); self.ed = ed
        self.t_mlp = nn.Sequential(nn.Linear(ed, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.c_mlp = nn.Sequential(nn.Linear(COND_DIM, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.null = nn.Parameter(torch.zeros(ed))
        self.inc = nn.Conv2d(1, 32, 3, 1, 1)
        self.d1 = ResBlock(32, 32, ed); self.p1 = nn.Conv2d(32, 32, 4, 2, 1)
        self.d2 = ResBlock(32, 64, ed); self.p2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.d3 = ResBlock(64, 128, ed); self.p3 = nn.Conv2d(128, 128, 4, 2, 1)
        self.m1 = ResBlock(128, 128, ed); self.m2 = ResBlock(128, 128, ed)
        self.u3 = nn.ConvTranspose2d(128, 128, 4, 2, 1); self.r3 = ResBlock(256, 64, ed)
        self.u2 = nn.ConvTranspose2d(64, 64, 4, 2, 1); self.r2 = ResBlock(128, 32, ed)
        self.u1 = nn.ConvTranspose2d(32, 32, 4, 2, 1); self.r1 = ResBlock(64, 32, ed)
        self.out = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, x, t, cond, drop=None):
        e = self.t_mlp(time_embed(t, self.ed))
        ce = self.c_mlp(cond)
        if drop is not None:
            ce = torch.where(drop[:, None], self.null.expand_as(ce), ce)
        e = e + ce
        h0 = self.inc(x); h1 = self.d1(h0, e)
        h2 = self.d2(self.p1(h1), e); h3 = self.d3(self.p2(h2), e)
        m = self.m2(self.m1(self.p3(h3), e), e)
        u = self.r3(torch.cat([self.u3(m), h3], 1), e)
        u = self.r2(torch.cat([self.u2(u), h2], 1), e)
        u = self.r1(torch.cat([self.u1(u), h1], 1), e)
        return self.out(u)

    def forward_uncond(self, x, t):
        b = x.size(0)
        return self.forward(x, t, torch.zeros(b, COND_DIM, device=x.device),
                            torch.ones(b, dtype=torch.bool, device=x.device))


def q_sample(x0, t, n):
    a = abar.to(x0.device)[t][:, None, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * n


def train(epochs, p_drop=0.12, lr=2e-4, batch=128, ema_decay=0.999):
    torch.manual_seed(0)
    ds = Cells(); dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = UNet().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = copy.deepcopy(model); [p.requires_grad_(False) for p in ema.parameters()]
    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"]); start = ck["epoch"]
    model.train()
    for ep in range(start + 1, start + epochs + 1):
        tot = 0.0
        for x, c in dl:
            x, c = x.to(DEVICE), c.to(DEVICE); b = x.size(0)
            t = torch.randint(0, T_STEPS, (b,), device=DEVICE)
            n = torch.randn_like(x)
            drop = torch.rand(b, device=DEVICE) < p_drop
            pred = model(q_sample(x, t, n), t, c, drop)
            loss = F.mse_loss(pred, n)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)
        print(f"epoch {ep:3d} loss {tot/len(dl):.4f}")
    torch.save({"state": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "epoch": start + epochs,
                "edges": ds.edges}, CKPT)
    print(f"saved at epoch {start + epochs} (edges {ds.edges})")


def make_cond(vf, stiff, edges=None):
    v = [float(vf), 0., 0., 0.]; v[1 + STIFF.index(stiff)] = 1.; return v


@torch.no_grad()
def ddim_sample(model, n, cond_vec, steps=50, guidance=2.0, seed=None, symmetrize=False):
    if seed is not None: torch.manual_seed(seed)
    model.eval(); ab = abar.to(DEVICE)
    ts = torch.linspace(T_STEPS - 1, 0, steps, dtype=torch.long, device=DEVICE)
    x = torch.randn(n, 1, IMG, IMG, device=DEVICE)
    c = torch.tensor(cond_vec, dtype=torch.float32, device=DEVICE).expand(n, COND_DIM)
    for i in range(steps):
        t = ts[i].expand(n)
        ec = model(x, t, c); eu = model.forward_uncond(x, t)
        eps = eu + guidance * (ec - eu)
        a_t = ab[ts[i]]
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        x = (ab[ts[i+1]].sqrt() * x0 + (1 - ab[ts[i+1]]).sqrt() * eps) if i < steps-1 else x0
    p = (x.cpu().numpy()[:, 0] + 1) / 2
    if symmetrize:
        p = 0.25 * (p + p[:, :, ::-1] + p[:, ::-1, :] + p[:, ::-1, ::-1])
    return (p >= 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--epochs", type=int, default=8)
    s = sub.add_parser("sample")
    s.add_argument("--vf", type=float, default=0.4)
    s.add_argument("--stiff", default="high", choices=STIFF)
    s.add_argument("-n", type=int, default=16); s.add_argument("--guidance", type=float, default=2.0)
    s.add_argument("--use-ema", action="store_true", default=True)
    args = ap.parse_args()
    if args.cmd == "train":
        train(epochs=args.epochs)
    else:
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        m = UNet().to(DEVICE); m.load_state_dict(ck["ema"])
        cells = ddim_sample(m, args.n, make_cond(args.vf, args.stiff),
                            guidance=args.guidance, seed=0)
        print("mean vf:", round(float(np.mean([c.mean() for c in cells])), 3))


if __name__ == "__main__":
    main()
