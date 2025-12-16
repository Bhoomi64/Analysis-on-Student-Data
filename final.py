# ============================================================
# STUDENT PERFORMANCE PREDICTION USING ML (FINAL – STABLE)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

sns.set(style="whitegrid")

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("python_project.csv")
df.columns = df.columns.str.strip()
df.info()
df.describe()
# ============================================================
# 2. TARGET VARIABLE (HIGH PERFORMER)
# ============================================================
df["CGPA"] = pd.to_numeric(df["CGPA"], errors="coerce")
df.dropna(subset=["CGPA"], inplace=True)
df["CGPA"] = df["CGPA"].clip(0, 10)

df["High_Performer"] = (df["CGPA"] >= 8.5).astype(int)

# ============================================================
# 3. EXPLORATORY DATA VISUALIZATIONS
# ============================================================

# Daily Study Hours vs CGPA
plt.figure(figsize=(7,5))
sns.scatterplot(
    x="daily study hours",
    y="CGPA",
    hue="High_Performer",
    data=df
)
plt.title("Daily Study Hours vs CGPA")
plt.show()

# AI Tool Usage per Week vs CGPA
plt.figure(figsize=(7,5))
sns.boxplot(
    x="Ai tool per week usuage",
    y="CGPA",
    data=df
)
plt.title("AI Tool Usage per Week vs CGPA")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(11,8))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Numerical Feature Correlation Heatmap")
plt.show()


# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ============================================================
# 5. FEATURES & TARGET
# ============================================================
X = df.drop(["CGPA", "High_Performer"], axis=1)
y = df["High_Performer"]

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
