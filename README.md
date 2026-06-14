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
- **Auxetic:** there is **no** auxetic control. The training data has essentially
  no negative-Poisson cells, so ν is measured and displayed but cannot be
  targeted. Genuine auxetic design needs finer re-entrant geometry or real
  topology-optimization data.
- Stiffness and Poisson are computed at 24×24 with a softened void for speed, so
  they are close approximations of the full-resolution solver, not exact.

## Model

- `model/ddpm2_ep25.onnx` — 8 MB, opset 17. Small U-Net (~2M params), 32×32.
- Inputs: `x [b,1,32,32]` float32, `t [b]` int64, `cond [b,4]` float32 where
  `cond = [volume_fraction, stiff_low, stiff_med, stiff_high]`.
- Outputs: `eps_c`, `eps_u` (conditional and unconditional noise). The browser
  does classifier-free guidance: `eps = eps_u + w·(eps_c − eps_u)`, default w = 3.
- Trained on 8,040 synthetic cells, EMA weights, checkpoint selected by
  controllability (not loss). Details in `model/EXPORT_NOTES.md`.

## Repository layout

```
index.html              in-browser diffusion app (loads the ONNX model)
procedural/index.html   procedural generator + same evaluator (no download)
model/
  ddpm2_ep25.onnx       trained diffusion model
  EXPORT_NOTES.md       interface, calibration map, measured controllability
scripts/
  generate_unitcells.py dataset generator
  metrics.py            evaluator (homogenization) used to label/score
  ddpm2.py              model definition + training (PyTorch)
```

## How it works

The ONNX graph is only the denoiser. The DDIM sampling loop (50 steps),
classifier-free guidance, thresholding, optional symmetry folding, and the
property evaluator all run in JavaScript. This mirrors the project's
generate-then-filter method: sample candidates, then keep the ones that match
the target by measurement.

## License

MIT. See `LICENSE`.
