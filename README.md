# Metamaterial Generator

Generate topology-like metamaterial unit cells with target properties, using a
conditional diffusion model that runs **in your browser**. You set a target
volume fraction and stiffness class; the model denoises new periodic unit cells,
and each one is scored live with real physics (exact volume fraction, periodic
connectivity, and a numerical homogenization for effective stiffness and
Poisson's ratio).

This is the browser front-end for a six-stage project: synthetic dataset →
physics evaluator → VAE baseline → conditional diffusion → validation.

## Live demo

Two pages:

- **`index.html`** — runs the trained diffusion model (`model/ddpm2_ep25.onnx`)
  in-browser via ONNX Runtime Web. Conditions on volume fraction + stiffness.
- **`procedural/index.html`** — no model download; generates candidates with a
  fast procedural generator and the same in-browser evaluator. Good as a fallback
  and for instant results.

### Running it

The model page fetches a local `.onnx` file, so it must be served over HTTP (a
`file://` page cannot fetch it). From the repo root:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/index.html
```

The procedural page works by opening the file directly, no server needed.

### Deploying (GitHub Pages)

Push to GitHub, then in the repo settings enable Pages from the `main` branch
root. The site will serve at `https://<user>.github.io/metamaterial-generator/`.
Both pages are static; no build step.

## What's verified

- The ONNX export reproduces the PyTorch model: per-step parity 2e-6, and the
  full sampling loop (numpy scheduler + ONNX) matches the PyTorch sampler at
  **0.9989 IoU**.
- The in-browser homogenizer was validated against the Python solver: a solid
  cell returns E = 1.000, ν = 0.300; a layered cell returns the Voigt/Reuss
  bounds (0.50 and ~0).
- The JavaScript diffusion schedule (timestep indices, alpha-bar) matches the
  Python schedule exactly.

**Not yet confirmed in a real browser:** the ONNX Runtime Web wiring (tensor
creation, WebGPU/WASM execution) could not be executed in the build sandbox.
The numerics are verified; please test the model page locally once and confirm
the backend loads (the status line reports `webgpu` or `wasm`). If WebGPU is
unavailable it falls back to WASM, which is slower but works.

## Controls and honest limits

- **Volume fraction:** calibrated and reliable in roughly **0.37–0.57** (the
  model's response is monotonic but compressed). The UI slider is limited to that
  range. The calibration map is in `model/EXPORT_NOTES.md`.
- **Stiffness class (low/medium/high):** the model separates low vs high
  effective modulus by about **2×** at a fixed volume fraction.
- **Auxetic (target ν):** the diffusion page has a target-ν slider. Negative
  targets are routed to a **second, retrained DDPM** — an auxetic specialist
  trained on 2,000 FEM-labeled rotating-squares cells (rigid squares joined by
  corner hinges), conditioned on [vf, ν]. A batch typically comes out ~50–60%
  measured-auxetic and the page ranks by homogenized ν against the target
  (best-of-batch usually lands within ~0.1–0.2 of targets down to ≈ −0.6;
  single cells as deep as ν ≈ −1 appear). Honest limits: in auxetic mode vf
  control is approximate (generated vf ≈ 0.55 ± 0.04), and the ν slider steers
  the *distribution*, with final numbers always measured, never assumed.
  - *Why a second model:* four retraining attempts (warm-started scalar-ν
    channel, from-scratch scalar-ν, TopoDiff-style spatial condition planes,
    and an auxetic-class one-hot) showed the small mixed model cannot switch
    geometry *family* from a conditioning channel — the channel only biases
    statistics. Within one family, conditioning is a smooth geometry mapping
    and works. Conditioning strength also **peaks mid-training and then
    decays** (the specialist peaked at epoch 120 of 240), so the checkpoint is
    selected by a controllability probe, not by loss — same lesson as v2.
  - *Procedural page — "Auxetic (target ν)":* the parametric rotating-squares
    family with exact vf control; measured ν down to about **−0.65**, most
    reliable at vf **0.40–0.55**. The dataset's original "reentrant" bowtie
    family, by contrast, measures ν ≈ +0.23 — auxetic-like by label only.
- Stiffness and Poisson are computed at 24×24 with a softened void for speed, so
  they are close approximations of the full-resolution solver, not exact.

## Models

Two DDPMs with the same U-Net backbone (~2M params, 32×32, opset 17, 8 MB each),
routed by the target ν:

- `model/ddpm2_ep25.onnx` — the standard model. `cond [b,4] =
  [volume_fraction, stiff_low, stiff_med, stiff_high]`. Trained on 8,040
  synthetic cells; EMA weights; epoch chosen by controllability.
- `model/ddpm_aux_ep120.onnx` — the auxetic specialist. `cond [b,2] =
  [volume_fraction, nu]`. Trained on 2,000 rotating-squares cells labeled by
  the same homogenizer (59–63% measured auxetic, ν down to ≈ −1); EMA
  weights; epoch 120 of 240 chosen by a controllability probe (ν conditioning
  peaks there and decays with further training). Measured inference: 0.38 s
  per design (RTX 5060 laptop GPU, PyTorch, 50 DDIM steps), 0.52 s on CPU.
- Both graphs: inputs `x [b,1,32,32]` f32, `t [b]` int64, `cond` f32; outputs
  `eps_c`, `eps_u`. The browser does classifier-free guidance
  `eps = eps_u + w·(eps_c − eps_u)`, default w = 3. Both exports verified
  bit-consistent with PyTorch (per-step ~3–5e-6; full 50-step loop IoU 1.0000
  with matched RNG). Details in `model/EXPORT_NOTES.md`.
- The in-browser evaluator now homogenizes at the native 32×32 grid (the old
  32→24 downsample distorted thin features such as the specialist's hinges).

## Repository layout

```
index.html              in-browser diffusion app (loads the ONNX models)
procedural/index.html   procedural generator + same evaluator (no download)
model/
  ddpm2_ep25.onnx       standard diffusion model (vf + stiffness)
  ddpm_aux_ep120.onnx   auxetic specialist (vf + nu, rotating-squares family)
  EXPORT_NOTES.md       interfaces, calibration maps, measured controllability
scripts/
  generate_unitcells.py dataset generator
  metrics.py            evaluator (homogenization) used to label/score
  ddpm2.py              model definition + training (PyTorch)
```

Training code for the auxetic specialist (dataset generator, trainer, probes,
ONNX export) lives in the `Diffusion Model Extension` stage folder:
`make_auxetic_dataset.py`, `ddpm_aux.py`, `export_onnx_aux.py`, and `ddpm3.py`
(the documented record of the failed mixed-model conditioning attempts).

## How it works

The ONNX graph is only the denoiser. The DDIM sampling loop (50 steps),
classifier-free guidance, thresholding, optional symmetry folding, and the
property evaluator all run in JavaScript. This mirrors the project's
generate-then-filter method: sample candidates, then keep the ones that match
the target by measurement.

## License

MIT. See `LICENSE`.
