# ONNX Export: ddpm_aux_ep120.onnx (auxetic specialist, vf + nu conditioning, EMA)

Dedicated auxetic DDPM trained on 2,000 rotating-squares cells (rigid squares
joined by corner hinges, drawn native at 32x32, FEM-labeled by the Stage-3
homogenizer; 63% measured auxetic, nu down to ~-1.0). Same U-Net backbone as
v2. EMA weights from epoch 120 of 240, selected by a controllability probe:
nu conditioning PEAKS at ep120 (67% auxetic at request -0.8, best-of-batch to
-1.7) and decays with further training (12% by ep240) - same
conditioning-vs-overtraining tradeoff v2 hit.

## Why a separate model
Four attempts to add auxetic control to the mixed 10,040-cell model failed:
warm-started scalar-nu channel, from-scratch scalar-nu, TopoDiff-style spatial
cond planes, and an auxetic-class one-hot. The small mixed model's global
conditioning biases statistics (vf, ~2x stiffness) but cannot switch geometry
FAMILY. Within the homogeneous rotating-squares family, conditioning is a
smooth geometry mapping and works. A v2-replica retrain in this environment
reproduced the shipped v2 vf response (0.366/0.444/0.533 for requests
0.2/0.45/0.7), ruling out environment/code differences.

## Graph interface
- Inputs:  x [b,1,32,32] f32, t [b] int64, cond [b,2] f32 = [vf, nu]
- Outputs: eps_c, eps_u. Browser CFG: eps = eps_u + w*(eps_c - eps_u), w = 3.
- Scheduler identical to v2 (DDIM, betas linspace(1e-4, 0.02, 400)).

## Verification
- Per-step parity vs PyTorch: 4.7e-6 (cond) / 5.2e-6 (uncond).
- Full 50-step DDIM loop (numpy scheduler + ONNX vs PyTorch, matched CPU RNG):
  mean IoU 1.0000.

## Measured behaviour (ep120, guidance 3, request vf 0.47, n=32)
- request nu -0.2 .. -1.0: 50-62% of cells measure nu < -0.01; median -0.02 to
  -0.15; best-of-batch typically -0.3 .. -0.8 (occasionally beyond -1).
- The browser passes the target nu through (clamped to [-1.0, -0.2]) and ranks
  the batch by measured nu vs target: best cell usually within 0.1-0.2 of
  targets down to about -0.6.
- vf control is WEAK in this model: generated vf ~0.55 +/- 0.04 regardless of
  request. The demo says so; rank-by-measurement still applies.
- Inference time (measured, 50 DDIM steps, batch 1): 0.38 s on an RTX 5060
  laptop GPU, 0.52 s CPU (PyTorch 2.11). Replaces the old ~4 s estimate.

# ONNX Export v2: ddpm2_ep25.onnx (vf + stiffness conditioning, EMA)

Re-export after retraining on 8,040 cells with two-property conditioning and EMA.
Exports the EMA weights from epoch 25, selected by controllability (not loss):
epoch 29 had lower loss but weaker conditioning.

## Verification
- Per-step parity vs PyTorch: 1.9e-6 (cond) / 1.4e-6 (uncond).
- Full DDIM loop (numpy scheduler + ONNX) vs PyTorch: 0.9989 mean IoU.
- Opset 17, 8.06 MB. GroupNorm/SiLU/time-embedding export correctly.

## Graph interface  (CHANGED from v1: cond is now 4-dim)
- Inputs:  x [b,1,32,32] f32, t [b] int64, cond [b,4] f32
- cond = [volume_fraction, stiff_low, stiff_med, stiff_high]
  (vf is a scalar in ~[0.2,0.7]; the three stiffness entries are a one-hot)
- Outputs: eps_c [b,1,32,32], eps_u [b,1,32,32]
- Browser does CFG: eps = eps_u + w*(eps_c - eps_u). Recommended guidance w = 3.0.

## Scheduler (unchanged): DDIM, 50 steps
betas = linspace(1e-4, 0.02, 400); abar = cumprod(1-betas).
Per step: run model -> CFG combine -> x0 = (x - sqrt(1-a)*eps)/sqrt(a), clamp [-1,1]
-> x = sqrt(a_prev)*x0 + sqrt(1-a_prev)*eps. Final step: x = x0. Threshold at 0.5.

## vf calibration (epoch 25, guidance 3) — ship this with the model
CAL_REQUEST  = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
CAL_ACHIEVED = [0.367, 0.400, 0.442, 0.488, 0.529, 0.567]
To hit a target achieved vf, interpolate-invert this map to get the request value.
Reliable target range is ~0.37-0.57 (the model's vf response is monotonic but
compressed); requests outside that saturate.

## Measured controllability (epoch 25, guidance 3, homogenized)
- vf tracking: requests 0.2/0.4/0.6 -> achieved 0.36/0.44/0.51 (monotonic).
- stiffness: low E ~0.044 vs high E ~0.089 (about 2x separation) at vf 0.45.
- connectivity: 22-24 of 24 samples periodically connected.

## Honest limits
- vf range is compressed (calibratable within ~0.37-0.57). Wider range needs a
  stronger conditioning signal or more training; loss-based overtraining instead
  WEAKENS conditioning (epoch 29 < epoch 25), so select by the controllability probe.
- No auxetic conditioning: the dataset has essentially no negative-Poisson cells,
  so that target still comes only from the JS rejection filter, and rarely succeeds.
