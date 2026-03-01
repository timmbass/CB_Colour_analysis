# Tiny Calibration Layer

This project uses a small post-hoc calibration block instead of a full ML model.

## Parameters

- `alpha`: blend between base and drape logits.
  - `z_mix = alpha * z_base + (1 - alpha) * z_drape`
- `bias`: per-season additive offsets (length 4).
- `gamma`: temperature for margin confidence.
  - `confidence = sigmoid(gamma * (top1 - top2))`

Final calibrated logits:

`z_cal = alpha * z_base + (1 - alpha) * z_drape + bias`

## Labeled CSV Format

Required columns:

- `image_id`
- `y_true` (0..3, season index order `[Spring, Summer, Autumn, Winter]`)
- `z_base_0..3`
- `z_drape_0..3`

## Fit Params

Run:

```bash
python3 tools/fit_tiny_calibration.py --input data/calibration_train.csv --output calibration_params.json
```

The script prints:

- best `alpha`
- learned `bias`
- chosen `gamma`
- training top1/top2 accuracy
- margin min/median/max summary

## Inference Behavior

If `calibration_params.json` exists, inference uses it; otherwise defaults are used:

- `alpha=0.5`
- `bias=[0,0,0,0]`
- `gamma=3.0`

Low-confidence mode:

- if `confidence < 0.65`, app shows top-2 seasons and both palettes plus tie-breaker guidance
- else app emphasizes top-1 season

