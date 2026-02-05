---
slug: 'rec'
title: 'Rec - Système de Recommandation d\'Animes'
stack: ['Python', 'Flask', 'PostgreSQL', 'Machine Learning', 'Docker']
tags: ['Web', 'API', 'Recommandation', 'Machine Learning', 'Anime']
category: 'Application Web'
cover: '/images/projects/rec/cover.png'
gallery:
  - '/images/projects/rec/screenshot1.png'
  - '/images/projects/rec/demo.mp4'
links:
  - label: 'GitHub'
    href: 'https://github.com/Youxise/Rec'
---

## Résumé

Rec est une application web permettant aux utilisateurs de découvrir et recevoir des recommandations d'animes personnalisées basées sur leurs préférences. Le système utilise deux algorithmes de recommandation complémentaires : le filtrage basé sur le contenu (Content-Based Filtering) et le filtrage collaboratif (Collaborative Filtering via FP-Growth). L'application aide les utilisateurs à trouver rapidement des animes similaires à ceux qu'ils ont appréciés.

## Contexte et Objectifs

Ce projet a été développé pour résoudre le problème de la découverte d'animes dans un catalogue vaste et croissant. Les objectifs principaux étaient de :

- Créer un système de recommandation hybride combinant plusieurs approches algorithmiques
- Fournir une interface utilisateur intuitive et responsive pour la recherche et la sélection d'animes
- Exploiter les métadonnées des animes (genres, synopsis, popularité, scores) pour générer des recommandations pertinentes
- Utiliser des techniques d'apprentissage automatique modernes, notamment les embeddings textuels via sentence-transformers
- Déployer l'application dans un environnement containerisé pour faciliter le déploiement et la scalabilité

## Fonctionnalités principales

**Recherche intelligente en temps réel** : Interface de recherche avec suggestions automatiques pendant la saisie, permettant aux utilisateurs de trouver rapidement des animes dans la base de données.

**Recommandations personnalisées hybrides** : Système combinant deux algorithmes complémentaires :
- Filtrage basé sur le contenu : Utilise des embeddings textuels (synopsis, genres) et des caractéristiques numériques (score, popularité, épisodes) avec une similarité cosinus
- Filtrage collaboratif : Exploite l'algorithme FP-Growth pour identifier des patterns d'association entre les animes basés sur les comportements des utilisateurs

**Sélection multiple** : Les utilisateurs peuvent sélectionner jusqu'à 5 animes pour affiner leurs recommandations, le système agrège alors les similarités pour produire des suggestions optimales.

**Exploration de la base de données** (à venir) : Fonctionnalité permettant de trier et filtrer les animes par score, popularité et date de sortie.

## Technologies utilisées

**Backend** : 
- Python 3.10
- Flask (framework web léger)
- SQLAlchemy (ORM pour la gestion de la base de données)
- psycopg2 (connecteur PostgreSQL)

**Machine Learning** :
- sentence-transformers (génération d'embeddings textuels pour l'analyse sémantique)
- scikit-learn (calcul de similarité cosinus)
- mlxtend (implémentation de FP-Growth pour le filtrage collaboratif)
- NumPy et Pandas (traitement des données)

**Base de données** : 
- PostgreSQL (stockage des métadonnées d'animes, transactions utilisateurs, et règles d'association)

**Frontend** : 
- HTML5, CSS3, JavaScript
- Bootstrap 5 (framework CSS pour le design responsive)
- jQuery (interactions dynamiques)
- SweetAlert2 (notifications utilisateur élégantes)

**DevOps** : 
- Docker (containerisation de l'application)
- GitHub Actions (CI/CD)

**Tests** :
- pytest (framework de tests unitaires et d'intégration)

## Résultats

L'application Rec démontre avec succès l'efficacité d'une approche hybride pour la recommandation d'animes. En combinant le filtrage basé sur le contenu avec le filtrage collaboratif, le système génère des recommandations diversifiées et pertinentes qui tiennent compte à la fois des caractéristiques intrinsèques des animes et des patterns de consommation des utilisateurs.

**Accomplissements** :
- Système de recommandation fonctionnel intégrant deux algorithmes complémentaires
- Interface utilisateur moderne et intuitive avec recherche en temps réel
- Architecture scalable avec containerisation Docker
- Base de données structurée avec tables optimisées (métadonnées, transactions, règles d'association)
- Pipeline de données incluant le scraping, le nettoyage et la génération d'embeddings
- Tests automatisés pour garantir la fiabilité des recommandations et des routes API

**Impact** :
Le projet démontre une maîtrise des concepts avancés de machine learning appliqués à un cas d'usage concret, ainsi qu'une compréhension approfondie des architectures web modernes. L'utilisation d'embeddings textuels via sentence-transformers montre une approche state-of-the-art pour l'analyse sémantique, tandis que l'algorithme FP-Growth capture efficacement les associations fréquentes dans les données utilisateur.

**Perspectives d'amélioration** :
- Déploiement de la fonctionnalité de découverte avec filtres avancés
- Ajout d'un système d'authentification utilisateur pour personnaliser davantage les recommandations
- Optimisation des performances avec mise en cache des embeddings et des calculs de similarité
- Extension de la base de données avec plus de métadonnées (studios, voix, adaptations)
