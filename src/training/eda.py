import pandas as pd
from src.database.db_config import engine


def load_data():
    query = "SELECT * FROM patient_churn;"
    df = pd.read_sql(query, engine)
    return df


if __name__ == "__main__":

    df = load_data()

    print("Dataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nClass Distribution:")
    print(df["Churned"].value_counts())
    print("\nClass Distribution Percentage:")
    print(df["Churned"].value_counts(normalize=True))
