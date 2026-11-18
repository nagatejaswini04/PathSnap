
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "reports" / "model_comparison" /
    "model_comparison_results.csv"
)

OUTPUT_DIR = ROOT / "visualizations" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

sns.set_theme(style="whitegrid")


# 1. Overall model performance
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Overall ML Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_model_comparison.png",
    dpi=300
)
plt.close()


# 2. F1-score comparison
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Model",
    y="F1 Score"
)

plt.title("F1-Score Comparison Across Models")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "f1_comparison.png",
    dpi=300
)
plt.close()


# 3. ROC-AUC comparison
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Model",
    y="ROC AUC"
)

plt.title("ROC-AUC Comparison Across Models")
plt.ylabel("ROC-AUC")
plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_auc_comparison.png",
    dpi=300
)
plt.close()


print("Model comparison visualisations generated successfully.")

for file in OUTPUT_DIR.glob("*.png"):
    print(file)
