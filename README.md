# Quantum-Ready Raman Spectral Classification via Data-Driven Synthetic Augmentation

A machine learning pipeline that augments real Raman spectral data with data-driven synthetic samples, reduces them via PCA into quantum-angle-encoded features (8 qubits, [0, π] scaling), and benchmarks classical and ensemble classifiers with 5-fold cross-validation.

---

## Overview

Raman spectroscopy datasets are often small and high-dimensional, which makes them hard to classify reliably and difficult to adapt for emerging quantum machine learning (QML) workflows. This project addresses both problems in a single pipeline:

1. **Real data ingestion** — loads a labeled Raman spectral dataset (14 classes) and performs a stratified train/test split to prevent data leakage.
2. **Data-driven synthetic augmentation** — estimates class-specific variation and noise-floor parameters *only* from the training split, then generates new synthetic spectra in log-space using real class-specific residual patterns (no arbitrary/hand-tuned noise).
3. **Quantum-ready feature encoding** — reduces the (synthetic + real) spectra to 8 principal components and rescales them to the [0, π] range, matching the input domain expected by quantum rotation gates (Rx, Ry, Rz) for angle encoding on an 8-qubit system.
4. **Classical/ensemble benchmarking** — trains and cross-validates multiple classifiers (Decision Tree, SVM, KNN, Logistic Regression, Random Forest, AdaBoost, CatBoost, Voting, and Stacking ensembles) on the quantum-encoded features, reporting Accuracy, Precision, Recall, F1, and AUC.
5. **Publication-style figure** — generates a single multi-panel graphical-abstract figure summarizing the full pipeline (schematic, real vs. synthetic spectra, PCA variance, 3D feature projection, model comparison, ROC curves).

> **Note:** This notebook prepares data in the exact numerical format required for quantum angle encoding (PCA → 8 features → [0, π] scaling), but the classification step itself uses classical scikit-learn/CatBoost models. It does not run circuits on a quantum simulator or quantum hardware — think of it as the classical preprocessing/benchmarking stage of a hybrid quantum-classical workflow.

---

## Pipeline Summary

```
Raw Raman Spectra
        │
        ▼
Stratified Train/Test Split (leakage-safe)
        │
        ▼
Data-Driven Parameter Estimation (train split only)
   - class-wise std in log-space
   - feature-level noise floor
        │
        ▼
Synthetic Raman Spectrum Generation (log-space residual sampling)
        │
        ▼
Cleaning (drop NaN / Inf samples)
        │
        ▼
PCA Reduction → 8 components
        │
        ▼
MinMax Scaling → [0, π]  (quantum angle-encoding ready)
        │
        ▼
Classical / Ensemble Classification (5-fold Stratified CV)
        │
        ▼
Metrics + ROC Curves + Publication Figure
```

---

## Key Features

- **Leakage-safe design** — synthetic data parameters are estimated exclusively from the training split; the held-out test set is never touched during augmentation.
- **Data-driven, not arbitrary, augmentation** — noise and class-variation parameters (`SIGMA_ALPHA`, `NOISE_FLOOR`) are derived mathematically from the real training spectra rather than fixed by hand.
- **Quantum encoding compatibility** — output features are scaled to [0, π], the standard input range for parameterized quantum rotation gates.
- **Broad classical baseline** — 9 models/ensembles evaluated with identical cross-validation folds for fair comparison.
- **Reproducible** — fixed random seeds throughout for consistent results.

---

## Repository Structure

```
.
├── raman_dataset_with_quanum.ipynb   # Main notebook (full pipeline)
├── README.md                         # This file
└── data/                             # (not included) place your CSVs here
    ├── Fourteenlabels_RamanData.csv
    └── Fourteenlabels_labels.csv
```

---

## Requirements

- Python 3.10+
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
- catboost

Install everything with:

```bash
pip install numpy pandas scikit-learn matplotlib joblib catboost
```

The notebook was originally developed in Google Colab and uses `google.colab.files.upload()` for interactive CSV upload. If running locally (e.g., Jupyter/VS Code), replace that cell with direct `pd.read_csv("path/to/file.csv")` calls.

---

## Dataset

The Raman spectral dataset used in this project is publicly available on Mendeley Data:

🔗 **[Raman Spectroscopy Dataset](https://data.mendeley.com/datasets/yk75sxk6p5/1)**

Download the dataset from the link above and place the feature/label CSV files in a `data/` folder before running the notebook.

## Data Format

- **Feature CSV** — no header; each row is one Raman spectrum, each column is the intensity at a given Raman shift position.
- **Label CSV** — one header row, two columns: `label_code`, `label_name`, aligned row-for-row with the feature CSV.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Place your feature and label CSV files in a `data/` folder (or update the file paths in the notebook).
3. Open the notebook:
   ```bash
   jupyter notebook raman_dataset_with_quanum.ipynb
   ```
4. Run all cells in order. The pipeline will:
   - Split and estimate parameters from real data
   - Generate synthetic spectra
   - Produce `Quantum_Raman_Data.csv` and `Quantum_Raman_Labels.csv`
   - Train and cross-validate all classifiers
   - Display the final summary table, ROC curves, and the combined publication figure

---

## Output

- `Quantum_Raman_Data.csv` — 8-feature, [0, π]-scaled, quantum-encoding-ready dataset
- `Quantum_Raman_Labels.csv` — corresponding class labels
- Console summary of Accuracy / Precision / Recall / F1 / AUC (mean ± std) per model
- Micro-averaged ROC curve plot
- Multi-panel publication/graphical-abstract figure

---

## Roadmap / Possible Extensions

- Integrate an actual quantum simulator (e.g., PennyLane or Qiskit) to run the angle-encoded features through a variational quantum classifier for a true hybrid quantum-classical comparison.
- Add hyperparameter tuning (grid/random search) for the ensemble models.
- Expand synthetic generation with additional augmentation strategies (e.g., peak-shift or baseline-drift simulation).

---

## License

Add a license of your choice (e.g., MIT) before publishing.

## Citation

If you use this pipeline in your research, please cite this repository.
