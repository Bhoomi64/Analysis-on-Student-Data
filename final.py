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
