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
