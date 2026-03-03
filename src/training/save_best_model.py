import pandas as pd
import joblib
import wandb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.database.db_config import engine
from src.training.preprocess import build_preprocessor
from src.training.feature_engineering import engineer_features


def load_data():
    query = "SELECT * FROM patient_churn;"
    df = pd.read_sql(query, engine)
    return df


if __name__ == "__main__":

    df = load_data()
    df = engineer_features(df)

    target = "Churned"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    param_grid = {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    joblib.dump(best_model, "models/best_model.joblib")

    print("Best model saved successfully.")

    # Log artifact to W&B
    wandb.init(project="patient_churn_mlops", name="Model_Registry")

    artifact = wandb.Artifact(
        "patient_churn_model",
        type="model"
    )

    artifact.add_file("models/best_model.joblib")

    wandb.log_artifact(artifact)
    wandb.finish()

    print("Model artifact logged to W&B.")
