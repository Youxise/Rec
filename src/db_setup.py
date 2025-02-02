# This file was created to generate all the tables of the database
import pandas as pd
import os
from sqlalchemy import create_engine

# Database connection parameters from environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "rec_db"

# Create SQLAlchemy engine
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, pool_recycle=3600)

print("Creating tables and inserting data...")

# Load CSV files into PostgreSQL
data_path = "data"  # Use relative paths
csv_files = {
    "anime_metadata": "Animes-Dataset-cleaned.csv",
    "anime_metadata_pp": "Animes-Dataset-preprocessed.csv",
    "anime_transactions": "Rating-Dataset.csv",
    "association_rules": "Association-rules.csv"
}

for table, filename in csv_files.items():
    file_path = os.path.join(data_path, filename)
    if os.path.exists(file_path):  # Ensure file exists
        df = pd.read_csv(file_path)
        print(df.columns)
        print(df)
        df.to_sql(table, engine, if_exists="replace", index=False)
        print(f"Inserted data into {table}")
    else:
        print(f"File {file_path} not found!")

print("Database setup complete.")
