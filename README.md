# Cotes du week-end

Page web qui affiche chaque week-end les 3 favoris les plus probables (cote 1,30–1,50) du samedi
et du dimanche dans les 5 grands championnats (Premier League, Liga, Serie A, Bundesliga, Ligue 1).

Données : football-data.co.uk (gratuit, sans clé API). Les cotes du week-end y apparaissent en général dès le jeudi.

## Mise en ligne

1. **GitHub** : crée un dépôt, puis envoie-y les 4 fichiers : `app.py`, `requirements.txt`, `render.yaml`, `README.md`.
2. **Render** : New + → *Web Service* → connecte ton dépôt GitHub.
   - Si Render lit `render.yaml`, tout est déjà configuré.
   - Sinon : Build Command `pip install -r requirements.txt`,
     Start Command `gunicorn app:app --workers 1 --threads 2 --timeout 180`, plan Free.
3. Attends la fin du déploiement : Render te donne un lien du type `https://cotes-weekend.onrender.com`.

## Utilisation

- `/` : la page avec les paris
- `/api/picks` : les mêmes données en JSON
- `/refresh` : force la mise à jour (sinon : toutes les 6 h)

Le plan gratuit de Render s'endort après 15 min sans visite : le premier chargement peut prendre 30 à 60 secondes.

## Limites

Outil d'analyse, pas de garantie de gains. Backtest 2025-26 : ROI +6,9 % ± 11 points, avantage non prouvé.
