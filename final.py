
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

df = pd.read_csv("python_project.csv")
df.columns = df.columns.str.strip()
df.info()
df.describe()

df["CGPA"] = pd.to_numeric(df["CGPA"], errors="coerce")
df.dropna(subset=["CGPA"], inplace=True)
df["CGPA"] = df["CGPA"].clip(0, 10)

df["High_Performer"] = (df["CGPA"] >= 8.5).astype(int)

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="daily study hours",
    y="CGPA",
    hue="High_Performer",
    data=df
)
plt.title("Daily Study Hours vs CGPA")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(
    x="Ai tool per week usuage",
    y="CGPA",
    data=df
)
plt.title("AI Tool Usage per Week vs CGPA")
plt.show()

plt.figure(figsize=(11,8))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Numerical Feature Correlation Heatmap")
plt.show()


le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(["CGPA", "High_Performer"], axis=1)
y = df["High_Performer"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\n{name}")
    print("Accuracy :", acc)
    print("Precision:", precision_score(y_test, preds))
    print("Recall   :", recall_score(y_test, preds))
    print("F1 Score :", f1_score(y_test, preds))

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=["Not Top", "Top Performer"],
        cmap="Blues"
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    return acc

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    ),
    "SVM": SVC(kernel="rbf", C=3, gamma="scale"),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42
    )
}

acc_results = {}

for name, model in models.items():
    acc_results[name] = evaluate_model(model, name)

# ============================================================
# 11. ACCURACY COMPARISON
# ============================================================
plt.figure(figsize=(9,5))
sns.barplot(x=list(acc_results.keys()), y=list(acc_results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xticks(rotation=30)
plt.show()
# ============================================================
# 12. BEST MODEL SELECTION (CORRECT)
# ============================================================
best_model_name = max(acc_results, key=acc_results.get)
best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

print("\n====== BEST MODEL ======")
print("Model   :", best_model_name)
print("Accuracy:", acc_results[best_model_name])

print("\nClassification Report:\n")
print(classification_report(y_test, best_preds))

if hasattr(best_model, "feature_importances_"):
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,6))
    sns.barplot(x="Importance", y="Feature", data=imp_df.head(10))
    plt.title(f"Top 10 Important Features ({best_model_name})")
    plt.show()
else:
    print(f"{best_model_name} does NOT support feature importance.")

tree_model = models["Decision Tree"]

plt.figure(figsize=(36,18), dpi=150)
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["Not Top", "Top Performer"],
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True
)
plt.title("Decision Tree Visualization (Clear Spacing)", fontsize=20)
plt.show()
print("\nModel Accuracies:")
for model, acc in acc_results.items():
    print(f"{model}: {acc*100:.2f}%")
