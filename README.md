# TD-PANL reference implementation

This bundle contains the implementation of Timely Decisions with Primal--dual
Alternating Neural Learning (TD-PANL), together with lightweight synthetic and
continuous-glucose-monitoring (CGM) examples from the paper.

The example defaults are execution smoke tests: they use one repetition, only
two operating points, and at most five training epochs. They check that the
data pipeline, model code, and result writers run; they are not reduced
reproductions of the paper's numerical results. Figures drawn from smoke
outputs can therefore be sparse or degenerate.

The full experiments were run in a separate cluster environment and are not
reproduced by these defaults. Code for external comparison methods is outside
the scope of this reference implementation.

## Layout

- `pyseqdx_pkg/`: installable Python package containing the proposed method.
- `example/`: synthetic and CGM smoke-test drivers and their output folders.
- `misc/`: R figure scripts and the cluster-only numerical true-front
  reference.

## Python setup

From this directory, install the package in editable mode:

```bash
python -m pip install -e ./pyseqdx_pkg
```

The exact production cluster environment has not been reconstructed. The
package metadata is therefore an initial dependency declaration rather than a
validated lock file. The code imports PyTorch, NumPy, pandas, Matplotlib,
SciPy, statsmodels, scikit-learn, tqdm, seaborn, and Plotly. Install compatible
versions, including a PyTorch build appropriate for the available hardware,
before running the examples.

### Optional dev container

The `.devcontainer/` CPU profile is provided. It has not been
tested from a clean build and is not a reconstruction of the production
cluster environment. It is intended only as a convenient starting point for
the Python smoke tests and ordinary R figure scripts. It does not install the
extra R packages or provide the compute resources needed by the cluster-only
`explicit_mu` calculation.

To try it, open this `code/` directory in a Dev Containers-compatible editor
and select **Reopen in Container**. The package is installed in editable mode
after the container is created.

## Synthetic smoke test

```bash
python example/drive_pareto.py --what gen_and_run
```

Results are written under `example/pareto_out/`.

## CGM data preparation and smoke test

The CGM data are not redistributed. After reviewing and accepting its terms,
download `SevereHypoDataset.zip` from the
[Jaeb public dataset page](https://public.jaeb.org/datasets/diabetes). The
study is titled *Severe Hypoglycemia in Older Adults with Type 1 Diabetes: A
Study to Identify Factors Associated with the Occurrence of Severe
Hypoglycemia in Older Adults with T1D*.

After installing `pyseqdx`, run:

```bash
python -m pyseqdx.data.prepare_cgm /path/to/SevereHypoDataset.zip
```

The command reads `Data Tables/BDataCGM.txt` and
`Data Tables/BPtRoster.txt` from the archive and writes these package data
files:

- `pyseqdx_pkg/pyseqdx/data/CGM.csv`: cleaned measurements and case/control
  labels;
- `pyseqdx_pkg/pyseqdx/data/cgm_interp_label.csv`: five-minute interpolated
  measurements and hypoglycemia-event labels.

A hypoglycemia event is defined as glucose below 60 mg/dL for at least 20
minutes. One participant from the original 201-person cohort was excluded.

Run the CGM smoke test after preparing the data:

```bash
python example/drive_cgm.py --what gen_and_run
```

Results are written under `example/cgm_out/`.

## Figure scripts

`misc/see_simu.R` and `misc/see_realexamples.R` are the original scripts used
to aggregate full experiment outputs and draw figures. Run them from this
directory; their paths are fixed to the current layout, and they write PDFs to
`misc/figures/`. They require R with `tidyverse` and the full cluster result
files.

`example/explain_cgm.ipynb` is retained because it generates the individual
trajectory figures in the manuscript's CGM section. Its generated
`example/cgm_traj_*.pdf` files are retained as reference outputs.

Running these scripts against smoke outputs checks only that PDFs can be
written. Sparse heatmaps and nearly empty summaries are expected, and those
PDFs should not be compared with the paper's figures.

## Cluster-only true-front reference

`misc/explicit_mu/` preserves the calculation used for the numerical `true`
curve in the simulation analysis. It is reference code, not part of the CPU
smoke test: the recorded run used a 714-task Slurm array and is not expected to
be practical on a local machine.

The retained calculation consists of:

- `pareto_true.sh`: Slurm job-array configuration;
- `compute_true.R`: per-task multiplier-grid evaluation;
- `explicit_mu_fn.R` and `recurse_fn.R`: numerical model and recursion helpers;
- `pareto_laga_grid_tpr0.9.csv`: the 714-row multiplier grid used by the array.

The Slurm script assumes it is submitted from `misc/`. Its R module name and
resource request record the original, cluster-specific environment. The R
code requires `argparser`, `tidyverse`, `cubature`, and `MASS`. Each task would
write `explicit_mu/pareto_true/res_part<ID>.RData`; those cluster outputs are
not included in this bundle.
