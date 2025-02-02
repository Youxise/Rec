import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import fpgrowth, association_rules
import psycopg2
import os

engine = psycopg2.connect(
        host="localhost",  # Usually 'localhost' or the actual database server address
        database="rec_db",  # The name of your PostgreSQL database
        user=os.getenv("DB_USER"),  # Your PostgreSQL username
        password=os.getenv("DB_PASSWORD")  # Your PostgreSQL password
    )

# ------------------------------------------------- Content Filtering Recommender ------------------------------------------------- #

class ContentRecommender:
    def __init__(self, engine, embeddings_path=r"src/embeddings.npy"):
        """
        Initialise le modèle basé sur le contenu.
        """
        self.db = engine
        # Charger les données et embeddings
        print("Chargement des données...")
        self.data = pd.read_sql("SELECT * FROM anime_metadata", self.db)
        self.text_features = np.load(embeddings_path)
        self.numeric_features = pd.read_sql("SELECT * FROM anime_metadata_pp", self.db)[["Episodes", "Popularity", "Score", "Demographic"]].fillna(0).values
    
    def get_combined_features(self):
        """
        Combine les embeddings textuels avec les autres caractéristiques.
        """
        return np.hstack([self.text_features, self.numeric_features])
    
    def get_recommendations(self, anime_ids, top_n=10, weights=(0.75, 0.25)):
        """
        Recommande des animes similaires pour plusieurs titres en entrée.
        """
        if isinstance(anime_ids, int):  # If a single ID is provided, wrap it in a list
            anime_ids = [anime_ids]
   
        # Obtenir les indices des animes demandés
        anime_indices = self.data[self.data["MAL_ID"].isin(anime_ids)].index
        
        # Combiner les caractéristiques
        print("Combinaison des embeddings et des colonnes numériques...")
        combined_features = self.get_combined_features()
        
        # Calculer la similarité pour chaque anime d'entrée
        print("Calcul de la similarité cosinus pour chaque anime...")
        text_similarities = cosine_similarity(
            combined_features[anime_indices, :self.text_features.shape[1]],
            combined_features[:, :self.text_features.shape[1]]
        )
        numeric_similarities = cosine_similarity(
            combined_features[anime_indices, self.text_features.shape[1]:],
            combined_features[:, self.text_features.shape[1]:]
        )
        
        # Combinaison pondérée des similarités
        total_similarities = (
            weights[0] * text_similarities + weights[1] * numeric_similarities
        )
        
        # Moyenne des similarités pour tous les animes d'entrée
        aggregated_similarity = total_similarities.mean(axis=0)
        
        # Obtenir les indices des animes les plus similaires
        similar_indices = aggregated_similarity.argsort()[-(top_n + len(anime_ids)):][::-1]
        similar_indices = [idx for idx in similar_indices if idx not in anime_indices][:top_n]
        
        recommended_ids_list = self.data.iloc[similar_indices]["MAL_ID"].tolist()
        
        return recommended_ids_list

# ------------------------------------------------- Collaborative Recommender ------------------------------------------------- #

class FPGrowthRecommender:
    def __init__(self, engine):
        """
        Initialise le système de recommandation collaborative.
        """
        self.db = engine
        print("Chargement des données...")
        self.ratings = pd.read_sql("SELECT * FROM anime_transactions", self.db)
        self.rules = pd.read_sql("SELECT * FROM association_rules", self.db)

        # Reconverting and filtering
        self.rules['antecedents'] = self.rules['antecedents'].apply(lambda x: set(eval(x)))
        self.rules['consequents'] = self.rules['consequents'].apply(lambda x: set(eval(x)))

        self.rules = self.rules[self.rules['antecedents'].apply(lambda x: len(x) == 1)]
        self.rules = self.rules[self.rules['consequents'].apply(lambda x: len(x) == 1)]
        print("Terminé.")

    def _rebuild_rules(self):
        """
        Attention : cette fonction a été utilisée pour construire les règles d'association. 
        Vous n'avez pas à la relancer du fait que les règles sont déjà affectés à l'attribut 'rules' 
        à l'initialisation du modèle.
        """

        # Prepare data for FP-Growth: create a user-item matrix (one-hot encoded)
        # Create a binary user-item matrix where 1 means interaction (e.g., rating > 0)
        # Create a new column to indicate that a user has rated an anime
        user_item_matrix = self.ratings.pivot_table(index='user_id', columns='MAL_ID', values='rating', aggfunc='count', fill_value=0)

        # Since FP-Growth requires binary values (1 or 0), we'll convert ratings > 0 to 1
        user_item_matrix = user_item_matrix.applymap(lambda x: 1 if x > 0 else 0)

        # Apply FP-Growth algorithm to find frequent itemsets
        frequent_itemsets = fpgrowth(user_item_matrix, min_support=0.001, use_colnames=True)

        # Eliminate singletons
        frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]

        # Generate association rules from the frequent itemsets
        if not pd.DataFrame(frequent_itemsets).empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0, num_itemsets=len(frequent_itemsets), support_only=True)
            rules = rules[['antecedents', 'consequents', 'support']]

            # Edits to permit save
            rules['antecedents'] = rules['antecedents'].apply(lambda x: str(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: str(x))

            # Save in DB
            rules[['antecedents', 'consequents', 'support']].to_sql('association_rules', self.db, if_exists='replace', index=False)

    def get_recommendations(self, anime_ids):
        """
        Recommends animes based on association rules.
        
        Parameters:
            rules (pd.DataFrame): The association rules dataframe with 'antecedents' and 'consequents' columns.
            anime_ids (int or list of int): The anime ID(s) to base the recommendations on.
            top_n (int): Number of recommendations to return.

        Returns:
            pd.DataFrame: DataFrame containing recommended animes, their confidence, and lift scores.
        """

        if isinstance(anime_ids, int):  # If a single ID is provided, wrap it in a list
            anime_ids = [anime_ids]
        
        similar_index = []

        for anime_id in anime_ids:
            for _, row in self.rules.iterrows():
                # Check if the given anime ID is in the antecedents
                if anime_id in row['antecedents']:
                    similar_index.append({
                        'consequents': row['consequents'],
                        'support': row['support']
                    })

        recommended_index_list = []
        # Convert recommendations to DataFrame
        similar_index_df = pd.DataFrame(similar_index)

        # Sort by confidence and lift
        if not similar_index_df.empty:
            similar_index_df = similar_index_df.sort_values(by="support", ascending=False) # sort
            similar_index = list(similar_index_df["consequents"]) # get the list of ids
            recommended_index_list = [next(iter(fset)) for fset in similar_index if fset] # clean the list of frozensets ids
            # Get common recommendations
            if len(anime_ids) > 1:
                recommended_index_list = find_duplicates(recommended_index_list)

        # Return the recommendations
        return recommended_index_list
    

# Function that retrieves duplicates in a list
def find_duplicates(input_list):
    # Use a set to track seen elements
    seen = set()
    # Use a set to track duplicates
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

# Combine the results from both models
def combine_results(list1, list2, data, k=10):
    # Use sets to keep track of unique elements
    unique_from_list1 = []
    unique_from_list2 = []
    seen = set()
    
    # FP-GROWTH results
    # Get k/2 unique elements from list1
    for item in list1:
        if item not in seen:
            unique_from_list1.append(item)
            seen.add(item)
        if len(unique_from_list1) == k // 2:
            break

    # Content results
    # Get the remaining unique elements from list2
    for item in list2:
        if item not in seen and item in list(data["MAL_ID"]):
            unique_from_list2.append(item)
            seen.add(item)
        if len(unique_from_list2) == k - len(unique_from_list1):
            break

    # Combine the results
    result = unique_from_list1 + unique_from_list2
    return result


# ATTENTION : l'exécution n'est pas obligatoire si vous avez déjà les tables nécessaires de la base de données 
# Celle-ci peut prendre beaucoup de temps

if __name__ == "__main__":
    
    recommender = FPGrowthRecommender(engine)
    recommender._rebuild_rules()

