import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from random import randint
from datetime import datetime
import re

# Attention : l'exécution peut prendre beaucoup de temps


# Fonction pour introduire un délai entre les requêtes
def delay_request():
    time.sleep(randint(1, 23))  # Délai aléatoire entre 1 et 4 secondes

# En-têtes pour simuler une requête provenant d'un navigateur
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

year = 1968 # Debut year | Année de début
seasons = ["winter", "spring", "summer", "fall"] # Saisons de sortie

current_year = datetime.now().year
# get the current day of the year
doy = datetime.today().timetuple().tm_yday

# "day of year" ranges for the northern hemisphere
spring = range(90, 181)
summer = range(181, 272)
fall = range(272, 365)
# winter = everything else

if doy in spring:
  current_season = 'winter'
elif doy in summer:
  current_season = 'spring'
elif doy in fall:
  current_season = 'summer'
else:
  current_year -= 1
  current_season = 'fall'


season = ""

# Liste pour stocker les données extraites
anime_data = []
loop_break = False

while loop_break == False:
    print(f"YEAR {year}")
    # Stop condition 1
    if year == current_year and season == current_season:
          loop_break = True
    for season in seasons:
        print(f"Scraping season {season}...")
       # Stop condition 2
        if year == current_year and season == current_season:
            loop_break = True
            break
        # Scraping

        url = f'https://myanimelist.net/anime/season/{year}/{season}'

        # Requête HTTP pour récupérer la page
        page = requests.get(url, headers=headers)
    
        if page.status_code != 200:
            print("Erreur de requête, arrêt du scraping.")
            break

        # Analyse du contenu de la page
        soup = BeautifulSoup(page.text, 'html.parser')

        anime_entries = []
        html_list = []
        # Trouve toutes les entrées d'anime
        for i in range(1,6):
                class_name = f"seasonal-anime-list js-seasonal-anime-list js-seasonal-anime-list-key-{i}"
                section = soup.find("div", class_=class_name)
                if section:
                    anime_entries.extend(section.find_all("div", class_=f"js-anime-category-producer seasonal-anime js-seasonal-anime js-anime-type-all js-anime-type-{i}"))

   
        # Si aucune entrée n'est trouvée, arrête la boucle
        if not anime_entries:
            print("Aucune entrée trouvée, fin du scraping.")
            delay_request()
            continue

        # Boucle pour extraire les données de chaque entrée d'anime
        for entry in anime_entries:
            #print(entry)
            # Extraction des informations
            title = entry.select_one("h2", class_="h2_anime_title").get_text(strip=True)

            infos = entry.select_one("div.prodsrc > div.info") if entry.select_one("div", class_="prodsrc") else None

            start_date = infos.find('span', class_='item').get_text(strip=True) if infos else None
            
            episode_text = infos.find_all('span')[2].get_text(strip=True)

            episodes = re.sub(r'[^\d]+', '', episode_text) if episode_text else None

            id = entry.find("div", class_="genres js-genre")["id"]

            synopsis = entry.select_one("div.synopsis.js-synopsis > p.preline").get_text(strip=True)
            #print(synopsis)

            properties = entry.select("div.properties > div.property")
            audience = None
            studio = None 
            theme = None
            if properties:
                for property in properties:
                    caption = property.select_one(".property > span.caption").get_text(strip=True)
                    caption_item = property.select_one(".property > span.item").get_text(strip=True)
                    match caption:
                        case "Studio":
                            studio = caption_item if caption_item != "Unknown" else None
                        case "Theme":
                            theme = caption_item if caption_item != "Unknown" else None
                        case "Demographic":
                            audience = caption_item if caption_item != "Unknown" else None
            
            genre = [span.get_text(strip=True) for span in entry.find_all('span', class_="genre")]
            #print(genre)

            score = entry.find("div", title="Score").get_text(strip=True)
            #print(score)

            popularity = entry.find("div", title="Members").get_text(strip=True)
            #print(popularity)

            # # Ajout des données dans la liste
            anime_data.append({
                "MAL_ID": id,
                "Title": title,
                "Release": start_date,
                "Genre": genre,
                "Synopsis": synopsis,
                "Score": score if score != "N/A" else None,
                "Episodes": episodes,
                "Popularity": popularity if popularity != "N/A" else None,
                "Demographic": audience,
                "Studio": studio,
                "Theme": theme
                # "Cover-URL": img
            })

            print(f"Données ajoutées pour : {title}")

        # Introduit un délai entre les requêtes pour éviter le blocage
        delay_request()

        # Incrémente l'index pour passer à la page suivante
    year += 1

        # Conversion de la liste en DataFrame
    df = pd.DataFrame(anime_data)

        # Sauvegarde dans un fichier CSV en ajoutant du contenu en écrasant l'existant
    df.to_csv(r"C:\Users\etudiant\Desktop\URCA M1 IA\URCA M1 IA\BDA\Rec\data\Animes-Dataset-raw.csv", mode='w', header=["MAL_ID", "Title",
                "Release",
                "Genre",
                "Synopsis",
                "Score",
                "Episodes",
                "Popularity",
                "Demographic",
                "Studio",
                "Theme"], index=False, encoding='utf-8')
        
    

       
       








