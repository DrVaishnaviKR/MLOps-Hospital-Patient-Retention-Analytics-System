import joblib
import pandas as pd
import wandb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV

from src.database.db_config import engine
from src.training.preprocess import build_preprocessor
from src.training.feature_engineering import engineer_features


def load_data():
    query = "SELECT * FROM patient_churn;"
    df = pd.read_sql(query, engine)
    return df


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }


if __name__ == "__main__":

    # --- Load & Engineer ---
    df = load_data()

    # Compute fixed reference date from training data
    ref_date = pd.to_datetime(
        df["Last_Interaction_Date"], dayfirst=True, errors="coerce"
    ).max() + pd.Timedelta(days=1)

    df = engineer_features(df, reference_date=ref_date)

    # Save reference date for inference consistency
    joblib.dump(ref_date, "models/reference_date.joblib")

    print("Class Distribution:")
    print(df["Churned"].value_counts(normalize=True))

    target = "Churned"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # Use imblearn Pipeline with SMOTE to handle class imbalance
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(
                random_state=42
            ))
        ]
    )

    # --- GRID SEARCH (fast) ---
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20],
        "classifier__min_samples_split": [2, 5]
    }

    wandb.init(project="patient_churn_mlops", name="GridSearch_SMOTE")

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=3, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    grid_metrics = evaluate(grid.best_estimator_, X_test, y_test)

    wandb.log({
        "method": "GridSearch",
        **grid_metrics,
        "best_params": str(grid.best_params_)
    })
    wandb.finish()
    print("GridSearch Metrics:", grid_metrics)
    print("GridSearch Best Params:", grid.best_params_)

    # --- RANDOM SEARCH (fast) ---
    from sklearn.model_selection import RandomizedSearchCV

    wandb.init(project="patient_churn_mlops", name="RandomSearch_SMOTE")

    random_search = RandomizedSearchCV(
        pipeline, param_grid,
        n_iter=6, cv=3, scoring="roc_auc",
        random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    random_metrics = evaluate(random_search.best_estimator_, X_test, y_test)

    wandb.log({
        "method": "RandomSearch",
        **random_metrics,
        "best_params": str(random_search.best_params_)
    })
    wandb.finish()
    print("RandomSearch Metrics:", random_metrics)

    # --- BAYESIAN SEARCH (fast) ---
    from skopt import BayesSearchCV
    from skopt.space import Integer

    wandb.init(project="patient_churn_mlops", name="BayesianSearch_SMOTE")

    bayes_params = {
        "classifier__n_estimators": Integer(100, 300),
        "classifier__max_depth": Integer(5, 25),
        "classifier__min_samples_split": Integer(2, 10)
    }

    bayes_search = BayesSearchCV(
        pipeline, bayes_params,
        n_iter=10, cv=3, scoring="roc_auc",
        random_state=42, n_jobs=-1
    )
    bayes_search.fit(X_train, y_train)
    bayes_metrics = evaluate(bayes_search.best_estimator_, X_test, y_test)

    wandb.log({
        "method": "BayesianSearch",
        **bayes_metrics,
        "best_params": str(bayes_search.best_params_)
    })
    wandb.finish()
    print("BayesianSearch Metrics:", bayes_metrics)

    # --- Pick the best overall model ---
    results = {
        "grid": (grid, grid_metrics),
        "random": (random_search, random_metrics),
        "bayes": (bayes_search, bayes_metrics),
    }

    best_name = max(results, key=lambda k: results[k][1]["roc_auc"])
    best_model = results[best_name][0].best_estimator_
    best_metrics = results[best_name][1]

    print(f"\nBest method: {best_name}")
    print(f"Best metrics: {best_metrics}")

    # Save best model
    joblib.dump(best_model, "models/best_model.joblib")
    print("Best model saved to models/best_model.joblib")

    # Log artifact to W&B
    wandb.init(project="patient_churn_mlops", name="Best_Model_Registry")
    artifact = wandb.Artifact("patient_churn_model", type="model")
    artifact.add_file("models/best_model.joblib")
    artifact.add_file("models/reference_date.joblib")
    wandb.log_artifact(artifact)
    wandb.log({"best_method": best_name, **best_metrics})
    wandb.finish()

    print("Model artifact logged to W&B.")
    print("All experiments completed successfully.")
