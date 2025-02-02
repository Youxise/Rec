from flask import Flask, render_template, request, jsonify
import pandas as pd
import psycopg2

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.models import ContentRecommender, FPGrowthRecommender, combine_results  # assuming models are in a separate file

app = Flask(__name__)

# Connection to DB
def create_db_connection():
    connection = psycopg2.connect(
        host="localhost",  # Usually 'localhost' or the actual database server address
        database="rec_db",  # The name of your PostgreSQL database
        user="postgres",  # Your PostgreSQL username
        password="youx123"  # Your PostgreSQL password
    )
    return connection

# Function to fetch all anime data from the database
def fetch_anime_data(table):
    conn = create_db_connection()
    query = f"SELECT * FROM {table};"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Initialize your recommender classes
engine = create_db_connection()  
anime_metadata = fetch_anime_data("anime_metadata")
recommender1 = ContentRecommender(engine=engine)
recommender2 = FPGrowthRecommender(engine=engine)

# Main front-end
@app.route('/')
def index():
    return render_template("index.html")

# Query suggestions
@app.route('/search_animes', methods=['GET'])
def search_animes():
    query = request.args.get('query')
    print("Received query:", query)  # Add this to check the input
    
    # Filter results based on query
    results = anime_metadata[anime_metadata['Title'].str.contains(query, case=False, na=False)]
    
    # Return the filtered results (only Name and Release)
    return jsonify(results[['Title', 'Release']].to_dict(orient='records'))

# Generate recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.get_json()
        if not data or 'titles' not in data:
            return jsonify({"error": "Invalid input"}), 400

        anime_titles = data['titles']
        # Check if all titles exist
        missing_titles = [title for title in anime_titles if title not in anime_metadata["Title"].values]
        if missing_titles:
            raise ValueError(f"Les titres suivants ne sont pas trouvés : {', '.join(missing_titles)}")
        
        # Obtenir les indices des animes demandés
        anime_ids = anime_metadata.loc[anime_metadata["Title"].isin(anime_titles), "MAL_ID"]

        # Apply recommendation models
        similar_anime_ids1 = recommender1.get_recommendations(anime_ids, top_n=10, weights=(0.8, 0.2))
        similar_anime_ids2 = recommender2.get_recommendations(anime_ids)

        # Combine the results of the models
        final_recommend_list = combine_results(similar_anime_ids2, similar_anime_ids1, anime_metadata, k=10)
        
        # Fetch anime details based on indices
        recommended_animes = anime_metadata[anime_metadata["MAL_ID"].isin(final_recommend_list)]

        return jsonify(recommended_animes.to_dict(orient='records')), 200

    except Exception as e:
        app.logger.error(f"Error processing recommendation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# @app.route('/discover_animes', methods=['GET'])
# def discover_animes():
#     sort_by = request.args.get('sort_by', 'Score')
#     filter_year = request.args.get('year')
#     query = f"SELECT * FROM anime_metadata WHERE YEAR(\"Release\") = {filter_year} ORDER BY {sort_by} DESC LIMIT 10"
#     results = pd.read_sql(query, engine)
#     return jsonify(results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
