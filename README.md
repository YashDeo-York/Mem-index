# Memorization Index

Official implementation and artifacts for the accompanying paper (arXiv submission pending).
The paper PDF is included in this repo: `data_copying.pdf`.

## Paper
- Title: A Calibrated Memorization Index (MI) for Detecting Training Data Leakage in Generative MRI Models
- Authors: Yash Deo$ ,Yan Jia,Toni Lassila,Victoria J Hodge,Alejandro F Frangi,Chenghao Qian,Siyuan Kang,Ibrahim Habli
- arXiv: *coming soon*

If you build on this work, please use the citation section at the bottom.

## Repository Layout
- `core/` - reusable pipeline logic (feature extraction, duplication sweeps, calibration).
- `models/` - SAM/MRI-CORE model code.
- `scripts/` - runnable entry points that replace the notebooks.
- `data/` - datasets (ignored by git).
- `outputs/` - results, tables, and saved features (ignored by git).
- `pretrained_weights/` - model checkpoints (ignored by git).

## Method Summary (high level)
This codebase implements a memorization index built from encoder features. The
pipeline extracts multi-layer representations, computes similarity scores under
duplication/augmentation settings, and calibrates scores against a bootstrap
null distribution to produce the final index.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install a compatible PyTorch build if needed (see https://pytorch.org).
3. Place the MRI-CORE checkpoint at `pretrained_weights/mri_foundation.pth`,
   or pass `--checkpoint` to the scripts.

Note: `cfg.py` is a minimal config stub required by the SAM-based model code.
If you have the original MRI-CORE config, you can replace it directly.

## Quickstart (Reproduce the Core Experiments)
1. Prepare data in the expected folders (see Data Layout).
2. Run the final pipeline script:
   ```bash
   python scripts/run_final_models.py --dataset brats --dup-levels 0.05,0.15,0.30 --output-dir outputs/results
   ```
3. The script writes a summary CSV to `outputs/results/`.

## Data Layout
Expected default locations (override with script flags if needed):
- BRATS NIfTI volumes: `data/brats/images/` (optional masks in `data/brats/mask/`)
- Knee PNG slices: `data/knee/images/`
- Spine NIfTI volumes: `data/spine_mri/`

## Running the Scripts

Exploratory (replacement for `data_pre.ipynb`):
```bash
python scripts/data_pre_exploration.py --max-files 40 --slice-range 20 60 --dup-fraction 0.2
```

Final pipeline (replacement for `final_models.ipynb`):
```bash
python scripts/run_final_models.py --dataset brats --dup-levels 0.05,0.15,0.30 --output-dir outputs/results
```

Knee dataset example:
```bash
python scripts/run_final_models.py --dataset knee --png-root data/knee/images --dup-levels 0.05,0.15
```

Spine dataset example:
```bash
python scripts/run_final_models.py --dataset spine --spine-root data/spine_mri --dup-levels 0.05,0.15
```

## Artifacts
- Paper PDF: `data_copying.pdf`
- Example result tables: `outputs/results/contaminated_sets_0.csv`, `outputs/results/contaminated_sets_10.csv`, `outputs/results/contaminated_sets_50.csv`

## Notes
- Large data, outputs, and weights are excluded via `.gitignore`.
- The core logic used for the paper lives in `core/memorization_pipeline.py`.

## Citation
If you use this code, please cite the accompanying paper:

```bibtex
TBD
```
