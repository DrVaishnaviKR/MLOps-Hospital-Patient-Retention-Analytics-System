import pandas as pd
from db_config import engine


def load_data_from_neon():
    query = "SELECT * FROM patient_churn;"
    df = pd.read_sql(query, engine)
    return df


if __name__ == "__main__":
    df = load_data_from_neon()

    print("First 5 rows:")
    print(df.head())

    print("\nTotal rows:", len(df))

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
