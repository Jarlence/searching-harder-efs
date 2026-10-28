# Restoring and analysing the research materials

## 1. Verify and restore

Download and extract the complete repository, then run from its root:

```sh
python verify_repository.py
python restore_materials.py --verify-only
python restore_materials.py --output restored
```

The restoration utility uses Python 3.9+ and the standard library. It checks archive parts, reconstructed archives, scientific members, and source files against `integrity_manifest.json`. Choose a new output directory. Saved collections are restored separately, and `restored/project` combines R22, X11c, and X12m in that order.

## 2. Prepare the scientific environment

Create a separate environment and working copy of `restored/project`. Use the versions in `requirements-lock.txt` and each collection's environment and freeze records. The original numerical stack used Python 3.8.20, NumPy 1.24.4, SciPy 1.10.1, pandas 2.0.3, and scikit-learn 1.3.2. X11c and X12m also record their CUDA/PyTorch environment.

Each collection's frozen source files and environment define its reproduction settings. The adjacent source directories provide browsing access; the restored collections retain the complete version-specific records. The Python 3.9+ restoration utility runs separately from the recorded Python 3.8 scientific environment.

## 3. Inspect the saved results

Within `restored/project`, start with:

| Study | Summary location |
|---|---|
| Strict inner-fold scaling, original benchmark | `results/r22_strict_scaling/REPORT.md` |
| Additional eight-data-set validation | `results/new_data_confirmation_gpu/REPORT.md` and `primary.csv` |
| Same-size follow-up | `results/new_data_samesize/REPORT.md` and its effect and comparison files |

The collections contain 48,510 completed jobs. Within-job checkpoints are repeated observations; statistical inference uses datasets as its primary unit. Collection-specific records identify the analysis samples and their exclusions.

## 4. Recompute summaries

Run analysis commands in the separate working copy because they write to `results/`:

```sh
python analysis/strict_scaling_r22.py
python analysis/new_data_confirmation_gpu.py --analyse
python analysis/new_data_samesize.py --analyse
```

These entry points validate the recorded source, data, and numerical environments. The strict-scaling analysis requires the original source datasets; the additional-data analyses enforce their recorded CUDA environment. Saved CSV/JSON summaries support inspection without running those scripts.

The other scripts in `analysis/` generate reporting-practice and control summaries. Historical figure and manuscript builders require manuscript assets outside this code-and-results repository. The journal submission files are maintained separately.

## 5. Rerun experiments

`experiments/` and `extras/` contain the experiment dispatchers. Full reruns use the source data, frozen code, recorded environment, and compute resources for each collection. Retrieve original-benchmark data with the supplied data-card and acquisition scripts. The eight additional UCI source copies and processed matrices are included with attribution in [DATA_SOURCES.md](DATA_SOURCES.md).
