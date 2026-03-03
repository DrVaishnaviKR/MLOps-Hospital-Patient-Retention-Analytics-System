from sqlalchemy import text
from db_config import engine

create_table_query = """
CREATE TABLE IF NOT EXISTS patient_churn (
    id SERIAL PRIMARY KEY
);
"""

with engine.connect() as connection:
    connection.execute(text(create_table_query))
    connection.commit()

print("Table created successfully")
