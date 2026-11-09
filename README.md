# Code, protocols and results

Code, protocols and result files for the manuscript *When Do Physical Bounds Improve Conformal Uncertainty in Neural
Latent-Variable Models? A Characterization With Environmental Applications*.

Absolute local paths recorded in result files are replaced by `<ROOT>`. The commit identifiers quoted in the
supplementary material (provenance table) refer to the version-control history of the project, which is
available on request.

## Layout

| Path | Content |
|---|---|
| `experiments/scrl/` | Models (encoder, latent families, decoders), conformal calibration, prediction-set constructions |
| `experiments/scripts/` | One script per experiment; each docstring states the design and, where applicable, the preregistered criteria |
| `experiments/tests/` | Unit tests (theory checks, including the numerical check of Theorem 2 in `test_shift_theorem.py`) |
| `experiments/results/` | Result files of every reported experiment (JSON / text) |
| `experiments/figures/paper/` | Figures of the paper and `figure_data.json` |
| `experiments/PLAN.md` | Dated preregistration log (protocols written before the corresponding runs; mostly in Chinese) |
| `experiments/RESULTS_MANIFEST.md` | Claim-by-claim record of results and evidence roles (in Chinese) |
| `experiments/PROVENANCE.md` | Commit chronology of the held-out ERA5 confirmation |
| `paper/generated/numbers.tex` | Every number in the paper as a LaTeX macro, each annotated with its source file and claim |

## Reproducing the numbers and figures

```
pip install -r experiments/requirements.txt
cd experiments
python scripts/make_figures.py        # figures/paper/*.pdf and figure_data.json
python scripts/make_numbers.py        # paper/generated/numbers.tex from the result files
python -m pytest tests -q             # unit tests
```

No number in the manuscript is typed by hand: each is a macro in `paper/generated/numbers.tex` produced by
`make_numbers.py` from the files in `experiments/results/`, with an assertion wherever the text depends on
the direction of a result.

## Re-running experiments

Model checkpoints (about 140 MB) and raw data (about 2 GB) are not included. All datasets are public (Daymet,
NOAA/NSIDC sea-ice CDR v6, ResOpsUS with the National Inventory of Dams, ERA5 retrieved through the
Open-Meteo archive, ESA Snow_cci, CryoClim); the
loaders in `experiments/scrl/data_*.py` document the expected files. Training is run by the scripts in
`experiments/scripts/` (for example `run_synthetic.py`, `o1.py`, `era5_confirm.py`); checkpoints can be made
available to the editor on request.

## Status

The manuscript is under review. The LaTeX source of the manuscript is not included in this repository; `paper/generated/` holds the number macros that `make_numbers.py` writes.
