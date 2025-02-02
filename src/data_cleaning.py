import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:youx123@localhost:5432/rec_db")

engine = psycopg2.connect(DATABASE_URL)

# Filter rating dataframe 
def filter_rating_db(df, df2):

    # Filter rating dataframe using the anime dataframe ids
    values_to_keep = df["MAL_ID"].values
    df_filtered = df2[df2['MAL_ID'].isin(values_to_keep)]

    # Filter bad ratings
    df_filtered = df_filtered[df_filtered['rating'] > 6]

    # Filtrer les utilisateurs peu actifs
    min_ratings_per_user = 10

    df_filtered = df_filtered.groupby('user_id').filter(
        lambda x: len(x) >= min_ratings_per_user
    )

    return df_filtered

# Handle missing values
def handle_missing_values(df):
    df["Score"].fillna(5, inplace=True)
    df["Episodes"].fillna(0, inplace=True)
    df["Demographic"].fillna("Unprovided.", inplace=True)
    df["Studio"].fillna("Unprovided.", inplace=True)

# Transform string numbers to real numbers
def numerify_string_numbers(df):
    # Define the mapping
    mp = {'K': '*1e3', 'M': '*1e6', 'B': '*1e9'}
    
    # Replace suffixes and evaluate each row independently
    df['Popularity'] = df['Popularity'].replace(mp.keys(), mp.values(), regex=True)
    df['Popularity'] = df['Popularity'].apply(pd.eval)

# Encode column with LabelEncoder
def encode_demographic(df_column):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df_column)

# Create a context column that contains all the textual infos of an anime
def create_context_column(df):
    df['Genre'] = df['Genre'].apply(ast.literal_eval)
    df["Genre_String"] = df["Genre"].apply(lambda x: ", ".join(x))
    # Fill NaN values in 'Theme' and 'Synopsis' with a string
    df["Theme"] = df["Theme"].fillna("Unprovided.")
    df["Synopsis"] = df["Synopsis"].fillna("Unprovided")
    
    df["Context"] = "Genre : " + df["Genre_String"] + ", " + df["Theme"] + ". Synopsis : " + df["Synopsis"]

def clean_popularity(value):
    if isinstance(value, list):
        # Flatten single-element lists, or replace others with a default value
        return value[0] if len(value) == 1 else None  # Replace `None` with 0 or mean if desired
    return value

# Normalize numerical features with Min-Max
def normalize_numerical_columns(df):
    df["Score"] = MinMaxScaler().fit_transform(np.array(df["Score"]).reshape(-1, 1)) 
    df["Episodes"] = MinMaxScaler().fit_transform(np.array(df["Episodes"]).reshape(-1, 1))
    df["Popularity"] = df["Popularity"].apply(clean_popularity)
    df["Popularity"] = MinMaxScaler().fit_transform(np.array(df["Popularity"]).reshape(-1, 1)) 
    df["Demographic"] = MinMaxScaler().fit_transform(np.array(df["Demographic"]).reshape(-1, 1))

# Generate embeddings for the context column
def generate_embeddings(df, output_filepath="embeddings", model_name="all-MiniLM-L6-v2"):
    """
    Génère des embeddings à partir des synopsis d'anime.
    """
    # Charger le modèle Sentence Transformer
    print(f"Chargement du modèle Sentence Transformer : {model_name}")
    model = SentenceTransformer(model_name)
    
    # Extraire les synopsis
    synopsis_list = df["Context"].fillna("").tolist()
    
    # Générer les embeddings
    print("Génération des embeddings...")
    embeddings = model.encode(synopsis_list, batch_size=32, show_progress_bar=True)
    
    # Sauvegarder les embeddings
    np.save(output_filepath, embeddings)
    print(f"Embeddings sauvegardés dans : {output_filepath}")

# Remove uneccesary columns 
def drop_unecessary_columns(df):
    df.drop(columns=['Studio', 'Genre', "Genre_String", "Synopsis", "Theme", "Release", "Title", "Context"], inplace=True)

# Save in DB
def save_to_postgresql(df, table_name):
    """
    Sauvegarde un DataFrame dans une table PostgreSQL.
    """
    print(f"Sauvegarde des données dans la table {table_name}...")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Données sauvegardées dans la table {table_name}.")


# ATTENTION : l'exécution n'est pas obligatoire si vous avez déjà les tables nécessaires de la base de données 
# Celle-ci peut prendre beaucoup de temps

if __name__ == "__main__":

    # Charger les données
    df = pd.read_csv(r"C:\Users\etudiant\Desktop\URCA M1 IA\URCA M1 IA\BDA\Rec\data\Animes-Dataset-raw.csv").drop_duplicates()
    df2 = pd.read_csv(r"C:\Users\etudiant\Desktop\URCA M1 IA\URCA M1 IA\BDA\Rec\data\\Rating-Dataset-raw.csv")

    # Prétraitement des données de notation
    print("Prétraitement des données de notation...")
    df2_filtered = filter_rating_db(df, df2)
    print(df2_filtered)
    
    save_to_postgresql(df2_filtered, "anime_transactions")
    print("Terminé !")

    # Prétraitement des données d'animes
    print("Prétraitement des données d'animes...")
    cleaned_data = df.copy()

    handle_missing_values(cleaned_data)     # Handle missing values
    numerify_string_numbers(cleaned_data)   # Transform K, M, B to real numbers
    print(cleaned_data)
    save_to_postgresql(cleaned_data, "anime_metadata")  # Sauvegarde dans PostgreSQL

    create_context_column(cleaned_data)     # Context = Genre + Theme + Synopsis
    cleaned_data["Demographic"] = encode_demographic(cleaned_data["Demographic"])
    normalize_numerical_columns(cleaned_data)         # Normalize columns using MinMax
    generate_embeddings(cleaned_data)       # Génération des embeddings
    drop_unecessary_columns(cleaned_data)   # Drop unecessary columns
    print(cleaned_data)
    save_to_postgresql(cleaned_data, "anime_metadata_pp")  # Sauvegarde dans PostgreSQL

    print("Terminé !")
