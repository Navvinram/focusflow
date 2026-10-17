"""Train and compare several regressors (productivity) and classifiers (distraction)."""
import joblib
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from data_generator import generate

FEATURES = ["hour", "day_of_week", "sleep_duration", "phone_usage",
            "stress", "energy", "motivation"]

# display name -> internal key (the order here is the order shown in the app)
REGRESSOR_NAMES = {
    "Linear Regression": "linear",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}
CLASSIFIER_NAMES = {
    "Logistic Regression": "logistic",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}


def _regressors():
    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=1),
    }


def _classifiers():
    return {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=1),
    }


def train_and_save_models(path="focusflow_models.joblib"):
    df = generate()
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(
        df[FEATURES], df["productivity_score"], df["distraction"],
        test_size=0.2, random_state=1)

    regressors, regression_metrics = {}, {}
    for key, model in _regressors().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        regressors[key] = model
        regression_metrics[key] = {
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": mean_squared_error(yte, pred) ** 0.5,
            "R2": r2_score(yte, pred),
        }

    classifiers, classification_metrics = {}, {}
    for key, model in _classifiers().items():
        model.fit(Xtr, dtr)
        pred = model.predict(Xte)
        classifiers[key] = model
        classification_metrics[key] = {
            "Accuracy": accuracy_score(dte, pred),
            "Precision": precision_score(dte, pred, zero_division=0),
            "Recall": recall_score(dte, pred, zero_division=0),
            "F1": f1_score(dte, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(dte, model.predict_proba(Xte)[:, 1]),
        }

    bundle = {"version": 2, "regressors": regressors, "classifiers": classifiers,
              "regression_metrics": regression_metrics,
              "classification_metrics": classification_metrics}
    joblib.dump(bundle, path)
    return bundle


if __name__ == "__main__":
    b = train_and_save_models()
    for k, m in b["regression_metrics"].items():
        print("regression", k, {a: round(v, 3) for a, v in m.items()})
    for k, m in b["classification_metrics"].items():
        print("classification", k, {a: round(v, 3) for a, v in m.items()})
