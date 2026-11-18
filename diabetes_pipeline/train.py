
import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/kaggle_diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. DEFINE MODELS AND PARAMETERS
# ==========================================

models = {

    "Logistic Regression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "params": {
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["liblinear", "lbfgs"]
        }
    },

    "K-Nearest Neighbors": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier())
        ]),
        "params": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"]
        }
    },

    "Support Vector Machine": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC())
        ]),
        "params": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"]
        }
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "criterion": ["gini", "entropy"]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [2, 3, 5]
        }
    }
}


# ==========================================
# 4. GRID SEARCH CV
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

best_model = None
best_model_name = None
best_score = -1

for name, config in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)
    print("=" * 50)

    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Best Parameters:", grid.best_params_)
    print("CV Accuracy:", grid.best_score_)
    print("Test Accuracy:", accuracy)

    results.append({
        "Model": name,
        "CV Accuracy": grid.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    if grid.best_score_ > best_score:

        best_score = grid.best_score_
        best_model = model
        best_model_name = name


# ==========================================
# 5. COMPARE ALL MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Accuracy",
    ascending=False
)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))


# ==========================================
# 6. SAVE RESULTS
# ==========================================

os.makedirs("experiments", exist_ok=True)

results_df.to_csv(
    "experiments/model_comparison.csv",
    index=False
)


# ==========================================
# 7. BEST MODEL EVALUATION
# ==========================================

print("\nBEST MODEL:", best_model_name)
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_model.predict(X_test)))


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

model_path = "experiments/best_diabetes_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(best_model, file)

print("\nBest model saved successfully!")
print("Model:", best_model_name)
print("Location:", model_path)
