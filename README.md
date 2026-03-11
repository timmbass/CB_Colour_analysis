# CB Colour Analysis

Streamlit app for personal colour analysis from uploaded portrait photos. The app detects facial regions, samples skin/hair/iris colour signals, scores seasonal palettes, renders digital drapes, and generates a PDF scorecard.

## Requirements

- Python 3.10+
- The MediaPipe face model at `assets/models/face_landmarker.task`

## Install

Runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Or with extras:

```bash
python -m pip install -e ".[dev]"
```

## Run

From any working directory:

```bash
streamlit run /absolute/path/to/CB_Colour_analysis/app.py
```

Or from the repo root:

```bash
streamlit run app.py
```

## Test

```bash
pytest
```

Smoke test:

```bash
python -m src.smoke_test
```

Lint and type check:

```bash
ruff check .
mypy src
```

## Project Structure

- `app.py`: thin Streamlit entrypoint
- `src/`: core analysis, config, reporting, and pure logic
- `ui/`: Streamlit-specific controls, charts, and presentation helpers
- `assets/`: palettes, copy rules, configs, and model assets
- `tests/`: unit and regression tests
- `tools/`: supporting scripts such as calibration fitting

## Key Assets

- MediaPipe model: `assets/models/face_landmarker.task`
- Palette data: `assets/palettes/seasonal_palettes.json`
- Copy rules: `assets/copy_rules.v1.json`
- Dynamic colour config: `dynamic_colors_config.json`
- Analysis thresholds/scoring config: `assets/config/analysis.v1.json`

## Notes

- Paths are resolved relative to the repository, so the app and tests work from a fresh clone without `PYTHONPATH=.`
- `calibration_params.json` is optional; defaults are used if it is absent
- The drape and season scoring pipeline is heuristic and intended as an MVP analysis workflow, not a calibrated medical or cosmetic diagnostic system

## Known Limitations

- Results depend heavily on lighting, image quality, and visible facial features
- The app currently assumes a single face per uploaded image
- The PDF scorecard is generated locally with matplotlib and may be slower on constrained servers
