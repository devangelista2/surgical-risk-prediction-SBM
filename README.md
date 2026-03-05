# MedModel Training Pipeline

A streamlined, robust, and highly configurable machine learning pipeline for medical tabular data. It handles both classification and regression automatically, supports complex temporal/predefined splitting strategies, trains Scikit-Learn and PyTorch models, and exports publication-ready scientific plots.

## 📂 Project Structure

```text
.
├── train.py                  # Main training execution script
├── preprocessing.py          # Custom scikit-learn transformers (dates, multilabel)
├── utils/
│   ├── logger.py             # Standardized terminal logging
│   └── vis.py                # Publication-ready plotting utilities (matplotlib/seaborn)
├── nn/
│   ├── torch_mlp.py          # PyTorch Multi-Layer Perceptron
│   └── torch_ft_transformer.py # PyTorch FT-Transformer
├── experiments/
│   └── train.sh              # Bash script for easy experiment configuration
├── data_config.json          # Maps the dataset features and target
└── parameters.json           # Defines hyperparameters for the models
```

## ⚙️ Configuration

### 1. `data_config.json`
Defines your dataset. Group your features appropriately so the pipeline knows how to scale and encode them.
```json
{
    "input_file": "data/SBM1212.xlsx",
    "input_features": ["Age", "Sex", "Pre-Op KPS", "Radio_Tumor side"],
    "cols_string": ["Sex", "Radio_Tumor side"],
    "cols_date": [],
    "cols_multi": []
}
```

### 2. `parameters.json`
Define the hyperparameters for any model you wish to use (`hgb`, `rf`, `lr`, `ridge`, `svc`, `torch_mlp`, `torch_ft_transformer`).

## 🚀 Usage

You can run the script directly via python:
```bash
python train.py --target "Severe_complication" --split_strategy temporal --date_column "Date of surgery"
```

**Or use the provided bash script for easier experiment management:**
```bash
cd experiments
./train.sh
```

## 🧠 Inference (Loading Saved Weights)

The script automatically saves the entire trained pipeline (imputers, scalers, encoders, and the model itself) as `pipeline.joblib`. 
To use this model on new, unseen patients later:

```python
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("benchmark_output/rf/pipeline.joblib")

# Load new patient data (must contain the same features defined in data_config.json)
new_patients = pd.read_csv("new_patients.csv")

# Predict directly! The pipeline handles all preprocessing internally.
predictions = pipeline.predict(new_patients)
probabilities = pipeline.predict_proba(new_patients)
```

---

## 📊 How to Read the Generated Plots

When training finishes, the output folder will contain a `metrics.json` file and several high-resolution (`300 DPI`) plots tailored for scientific publication.

### 1. Confusion Matrix (`confusion_matrix.png`)
* **What it shows:** A grid comparing the *Actual* patient outcomes (True Label) against the *Predicted* outcomes by the model.
* **How to read it:** * **Diagonal cells** (top-left to bottom-right) represent correct predictions (True Positives and True Negatives). 
  * **Off-diagonal cells** represent errors (False Positives and False Negatives). In clinical settings, predicting a complication when there isn't one (False Positive) is usually preferred over missing a fatal complication (False Negative).

### 2. ROC Curve (`roc_curve.png`)
* **What it shows:** The trade-off between the True Positive Rate (Sensitivity) and the False Positive Rate (1 - Specificity) across different probability thresholds.
* **How to read it:** * The dashed diagonal line represents random guessing (AUC = 0.50).
  * The closer the solid curve gets to the top-left corner, the better the model is at distinguishing between the two classes. 
  * **AUC (Area Under the Curve):** A value of 1.0 means perfect separation. A value > 0.80 is generally considered excellent for clinical models.

### 3. Precision-Recall Curve (`pr_curve.png`)
* **What it shows:** The trade-off between Precision (Positive Predictive Value) and Recall (Sensitivity). 
* **How to read it:** This plot is highly recommended over the ROC curve when your dataset is **imbalanced** (e.g., only 5% of patients have the complication). A model that stays close to the top-right corner is highly effective at finding the rare minority class without throwing too many false alarms.

### 4. Actual vs Predicted Plot (`actual_vs_predicted.png`)
* **What it shows:** Used *only* for regression tasks (e.g., predicting "Days of hospitalization"). It plots the model's prediction on the Y-axis against the actual truth on the X-axis.
* **How to read it:** * The red dashed line represents perfect prediction ($y = x$). 
  * Points clustered tightly along this line indicate high accuracy. 
  * If points fan out heavily at higher values, the model is struggling to predict extreme/high outcomes.