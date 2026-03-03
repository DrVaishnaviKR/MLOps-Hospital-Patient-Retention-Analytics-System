import pandas as pd
from db_config import engine

df = pd.read_csv("data/raw/patient_churn_dataset.csv")

df.to_sql(
    name="patient_churn",
    con=engine,
    if_exists="replace",
    index=False
)

print("Dataset uploaded successfully to Neon")
