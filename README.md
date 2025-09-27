# Emerge123456

Emerge is a Python toolkit for harmonising single-cell RNA sequencing (scRNA-seq) data with
spatially resolved FISH-like transcriptomic measurements. It aligns cell types across
modalities, reconstructs high-resolution gene expression maps for spatial data, and quantifies
local micro-environmental features using optimal transport.

Emerge exposes a single high-level entry point – [`run_emerge`](emerge/emerge.py) – that takes
pandas `DataFrame` objects describing expression matrices, cell metadata and spatial
coordinates and returns harmonised predictions for genes that are missing in the spatial
measurement platform.

## Key capabilities

- **Cross-modality mapping** – integrates scRNA-seq reference profiles with spatial FISH data to
  transfer cell-type annotations.
- **Environment-aware optimal transport** – combines spatial neighbourhood features with
  expression similarity to match cells between datasets.
- **Gene expression imputation** – predicts the expression of genes that are not directly
  measured in the spatial modality.
- **Flexible backends** – supports both standard and large-scale neighbourhood extraction
  strategies depending on dataset size.

## Installation

Emerge targets Python 3.9 or newer. The recommended way to install the package is via `pip`
from a local clone:

```bash
# Clone the repository
git clone https://github.com/your/repo.git
cd Emerge

# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

The installation pulls in the required scientific Python stack, including `numpy`, `pandas`,
`scanpy`, `anndata`, `scikit-learn` and the POT optimal transport library. If you plan to run
the tutorial notebooks, ensure that Jupyter is available in your environment as well.

## Preparing your data

`run_emerge` operates on pre-loaded `pandas.DataFrame` objects. The minimal inputs are:

| Argument | Description |
| --- | --- |
| `fish_exp_raw` | Spatial expression matrix with cells in rows and genes in columns. |
| `fish_type` | *(Optional)* DataFrame whose index matches `fish_exp_raw` and contains a `cell_types` column with prior labels. Missing values are filled with `"Unknown"`. |
| `fish_loc` | DataFrame containing spatial coordinates for the cells; columns default to `"X"` and `"Y"`. |
| `scrna_sc_raw` | scRNA-seq expression matrix (cells × genes) used as the reference atlas. |
| `scrna_type_raw` | DataFrame describing scRNA-seq metadata with a `layeruse` column (defaults to `"Annotation"`) that stores cell-type annotations. |

All DataFrames must share a consistent set of gene names so that overlapping genes between
`fish_exp_raw` and `scrna_sc_raw` can be identified. Gene names present only in the scRNA-seq
reference can be supplied via the `test_genes` argument to control which targets are imputed.

## Quick start

```python
import pandas as pd
from emerge import run_emerge

# Load your data (replace with actual file paths)
fish_exp = pd.read_csv("fish_expression.csv", index_col=0)
fish_metadata = pd.read_csv("fish_metadata.csv", index_col=0)
fish_locations = fish_metadata[["X", "Y"]]
sc_exp = pd.read_csv("sc_expression.csv", index_col=0)
sc_metadata = pd.read_csv("sc_metadata.csv", index_col=0)

results = run_emerge(
    fish_exp_raw=fish_exp,
    fish_type=fish_metadata[["cell_types"]],
    fish_loc=fish_locations,
    scrna_sc_raw=sc_exp,
    scrna_type_raw=sc_metadata,
    type_col="cell_types",
    layeruse="Annotation",
)

predicted_expression = results["pre_sc_test"]  # DataFrame: spatial cells × imputed genes
```

The resulting dictionary also contains diagnostic information about the matching process (see
[Outputs](#outputs)).

## Parameter overview

The function signature exposes several optional knobs for fine-tuning the algorithm:

- `train_genes` / `test_genes`: control which genes are used for alignment and which are
  predicted. Defaults to all shared genes for training and all scRNA-only genes for testing.
- `k_label_list`: candidate numbers of nearest neighbours when transferring labels from the
  scRNA reference to the spatial data.
- `k_knn_list`: neighbour counts used while constructing optimal transport couplings.
- `mruse`: collection of distance metrics (from scikit-learn / POT) used to build
  multi-metric averages for robustness.
- `type_threod`: confidence threshold for keeping predicted cell types before back-filling
  low-confidence cells via k-nearest neighbours.
- `alpha_linear`, `epsilon`, `tol`, `max_iter`: parameters of the entropic optimal transport
  solver controlling the balance between expression and environmental distances.
- `env_backend`: choose `"standard"` (PCA-based aggregation) or `"big"` (BallTree-based
  neighbourhood processing) depending on dataset size. Adjust `big_env_processes` to tune
  parallelism when using the big-data backend.
- `cell_or_cluster`: decide whether annotations are transferred per cell or per cluster.

Refer to the docstring of [`run_emerge`](emerge/emerge.py) for the full list of parameters.

## Outputs

`run_emerge` returns a dictionary with the following keys:

| Key | Type | Description |
| --- | --- | --- |
| `pre_sc_test` | `pd.DataFrame` | Predicted expression matrix for the requested `test_genes`. |
| `type_index_list_cross` | `List[List[int]]` | Indices of the selected environment model for each cell type across distance scales. |
| `type_stay_list_cross` | `List[List[str]]` | Cell types retained after applying the confidence threshold. |
| `eva_type_use_list_cross` | `List[List[pd.DataFrame]]` | Environment feature matrices used for optimal transport per cell type. |
| `weighted_average_types_cross` | `List[pd.DataFrame]` | Averaged probability scores for transferred cell-type labels. |
| `eva_type_use_list_cross_nos` | `List` | Reserved for additional diagnostics (currently empty). |

These artefacts can be used to inspect the alignment quality or debug parameter choices.

## Tutorials

Two Jupyter notebooks are bundled with the repository:

- `emerge_mop_un.ipynb` – end-to-end example of running Emerge on matched scRNA/FISH data.
- `mop_cross_validation.ipynb` – demonstrates cross-validation for gene prediction accuracy.

Launch Jupyter and open the notebooks to explore the workflow interactively:

```bash
jupyter notebook
```

## Citation

If you use Emerge in your research, please cite the associated manuscript (preprint coming
soon). In the meantime you can cite this repository:

> Yang, S.-T. & Zhang, X.-F. (2024). *Emerge: Environment-aware multimodal gene expression
> reconstruction*. GitHub repository. https://github.com/your/repo

## Contact

Please contact Miss **Yang Shi-Tong** (<styang@mails.ccnu.edu.cn>) or Dr. **Xiao-Fei Zhang**
(<zhangxf@mail.ccnu.edu.cn>) for questions about the repository and the algorithm.

## License

Emerge is released under the MIT License.

## Contributing

Bug reports, feature suggestions and pull requests are welcome. Please open an issue to discuss
major changes beforehand, and ensure new code paths are covered by tests or example notebooks
where applicable.
