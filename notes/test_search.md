Le champ `location_name` n'est jamais renseigné (N/A), Ça veut dire que cette métadonnée n'est pas remplie dans tes données sources, ou qu'elle se perd quelque part. Ce n'est pas bloquant (le lieu est dans location_city et dans le page_content), mais c'est à vérifier.

Les résultats sont globalement corrects sémentiquement, c’est le comportement attendu pour une recherche vectorielle. Elle capte le sens global mais pas les critères exacts (la ville précise, le genre musical exact, ...)






(puls-events) PS C:\Users\jonat\Desktop\Cours en ligne\Formation AI Engineer\Projets\07\puls-events> uv run python scripts/test_search.py
Index chargé : 1403 vecteurs

============================================================
REQUÊTE : concert jazz Valence
============================================================

 --- Résultats 1---
  Titre : Forró à Valence
  Lieu  : N/A — Bourg-lès-Valence
  Dates : 2024-09-18T16:00:00+00:00 → 2025-06-25T16:00:00+00:00
 Affichage dates : 18 septembre 2024 - 25 juin 2025, les mercredis
  Texte : Titre : Forró à Valence
Date : 18 septembre 2024 - 25 juin 2025, les mercredis
Lieu : Mjc Jean Moulin Bourg-lès-Valence, 20, Avenue Jean Moulin, 265...

 --- Résultats 2---
  Titre : Soirée cabaret
  Lieu  : N/A — Valence
  Dates : 2025-05-17T17:00:00+00:00 → 2025-05-17T17:00:00+00:00
 Affichage dates : Samedi 17 mai, 19h00
  Texte : Titre : Soirée cabaret
Date : Samedi 17 mai, 19h00
Lieu : Musée de Valence, art et archéologie, 4 Place des Ormeaux, 26000 Valence, France...

 --- Résultats 3---
  Titre : Forró à Bourg-les-Valence
  Lieu  : N/A — Bourg-lès-Valence
  Dates : 2025-09-10T17:00:00+00:00 → 2026-06-24T17:00:00+00:00
 Affichage dates : 10 septembre 2025 - 24 juin 2026, les mercredis
  Texte : Mots-clés : danse, forró, cours, Valence...

============================================================
REQUÊTE : exposition gratuite
============================================================

 --- Résultats 1---
  Titre : Visite libre de l'église
  Lieu  : N/A — La Roche-sur-Grane
  Dates : 2025-09-20T08:00:00+00:00 → 2025-09-21T12:00:00+00:00
 Affichage dates : 20 et 21 septembre
  Texte : Titre : Visite libre de l'église
Date : 20 et 21 septembre
...u : Eglise Saint-Jacques et Saint-Christophe, rue du village 26400 La-Roche-sur-Grane

 --- Résultats 2---
  Titre : SUR LES TRACES DU CASTOR
  Lieu  : N/A — Crest
  Dates : 2025-06-11T08:00:00+00:00 → 2025-07-05T08:00:00+00:00
 Affichage dates : 11 juin - 5 juillet
  Texte : Mots-clés : Exposition photo, Conférence, Ballades découvertes...

 --- Résultats 3---
  Titre : Visites éclairages de l'exposition "L'Arménie du sacré à l'épreuve du temps"
  Lieu  : N/A — Valence
  Dates : 2025-09-20T09:00:00+00:00 → 2025-09-21T15:00:00+00:00
 Affichage dates : 20 et 21 septembre 2025
  Texte : Titre : Visites éclairages de l'exposition "L'Arménie du sacré à l'épreuve du temps"
Date : 20 et 21 septembre 2025
Lieu : CPA, 14 Rue Louis Gallet,...

============================================================
REQUÊTE : spectacle enfants Montélimar
============================================================

 --- Résultats 1---
  Titre : Une histoire d'enfants
  Lieu  : N/A — Valence
  Dates : 2025-07-07T07:00:00+00:00 → 2025-07-25T07:00:00+00:00
 Affichage dates : 7 - 25 juillet
  Texte : Titre : Une histoire d'enfants
Date : 7 - 25 juillet
Lieu : Maison Pour Tous Petit Charran, 30 Rue Henri Dunant, 26000 Valence...

 --- Résultats 2---
  Titre : Poucette - Festival La Fête des Enfants - Sainte Croix (26)
  Lieu  : N/A — Sainte-Croix
  Dates : 2025-06-21T14:00:00+00:00 → 2025-06-21T14:00:00+00:00
 Affichage dates : Samedi 21 juin, 16h00
  Texte : Titre : Poucette - Festival La Fête des Enfants - Sainte Croix (26)
Date : Samedi 21 juin, 16h00
Lieu : Monastère de Sainte Croix, 54 place de l'Egl...

 --- Résultats 3---
  Titre : Zoom sur une architecture d'exception au château de Montélimar
  Lieu  : N/A — Montélimar
  Dates : 2025-09-20T09:30:00+00:00 → 2025-09-21T14:00:00+00:00
 Affichage dates : 20 et 21 septembre 2025
  Texte : Titre : Zoom sur une architecture d'exception au château de Montélimar
Date : 20 et 21 septembre 2025
Lieu : Château de Montélimar, Rue du Château, ...

============================================================
REQUÊTE : festival été Drôme
============================================================

 --- Résultats 1---
  Titre : Les Baudrières - Cirque vertical, musical et verbal
  Lieu  : N/A — La Chapelle-en-Vercors
  Dates : 2025-07-28T16:30:00+00:00 → 2025-07-28T16:30:00+00:00
 Affichage dates : Lundi 28 juillet, 18h30
  Texte : . Entre technique acrobatique et recherche chorégraphique sur cordes lisses, entre danse, jeu et musique live, ce spectacle ouvre un univers sensible ...

 --- Résultats 2---
  Titre : Concert "Building"
  Lieu  : N/A — Châtillon-en-Diois
  Dates : 2025-07-07T18:00:00+00:00 → 2025-07-07T18:00:00+00:00
 Affichage dates : Lundi 7 juillet, 20h00
  Texte : Titre : Concert "Building"
Date : Lundi 7 juillet, 20h00
Lieu : Camping municipal les Chaussières, 26410 Chatillon en diois
Description : L'Orchest...

 --- Résultats 3---
  Titre : Concert "Building"
  Lieu  : N/A — Saoû
  Dates : 2025-07-06T09:00:00+00:00 → 2025-07-06T09:00:00+00:00
 Affichage dates : Dimanche 6 juillet, 11h00
  Texte : Titre : Concert "Building"
Date : Dimanche 6 juillet, 11h00
Lieu : Ecole de Saoû, Route de la forêt, 26400 Saoû
Description : L'Orchestre de Chambr...

============================================================
REQUÊTE : atelier peinture
============================================================

 --- Résultats 1---
  Titre : Une histoire d'enfants
  Lieu  : N/A — Valence
  Dates : 2025-07-07T07:00:00+00:00 → 2025-07-25T07:00:00+00:00
 Affichage dates : 7 - 25 juillet
  Texte : . L’intention est également de leur faire prendre conscience de la richesse des images quand elles sont pensées, construites et réfléchies, à l’opposé...

 --- Résultats 2---
  Titre : De pierre et d'or - Œuvre collective
  Lieu  : N/A — Valence
  Dates : 2025-09-20T09:00:00+00:00 → 2025-09-21T09:00:00+00:00
 Affichage dates : 20 et 21 septembre 2025
  Texte : Titre : De pierre et d'or - Œuvre collective
Date : 20 et 21 septembre 2025
Lieu : CPA, 14 Rue Louis Gallet, 26000 Valence, France
Description : À ...

 --- Résultats 3---
  Titre : Exposition : Fany MORAND
  Lieu  : N/A — Saint-Christophe-et-le-Laris
  Dates : 2025-09-20T08:00:00+00:00 → 2025-09-21T08:00:00+00:00
 Affichage dates : 20 et 21 septembre
  Texte : . Et quand me suis-je mise au dessin numérique ? À l’arrivée de mes enfants ! Si j’étais encore bien crayons, pinceaux jusque-là (et j’avais déjà ma p...

============================================================
REQUÊTE : randonnée nature
============================================================

 --- Résultats 1---
  Titre : Sortie ''Faune sauvage''
  Lieu  : N/A — Bouvante
  Dates : 2025-04-15T09:00:00+00:00 → 2025-08-29T09:00:00+00:00
 Affichage dates : 15 avril - 29 août 2025
  Texte : Mots-clés : Nature, Sortie nature, biodiversité, photographie, mammifères, oiseaux...

 --- Résultats 2---
  Titre : Randonnée aquatique nature
  Lieu  : N/A — Bouvante
  Dates : 2025-05-09T09:00:00+00:00 → 2025-06-07T09:00:00+00:00
 Affichage dates : 9 mai - 7 juin 2025
  Texte : Titre : Randonnée aquatique nature
Date : 9 mai - 7 juin 2025
Lieu : Chem. de la Plaine, 26190 Bouvante, France, Chem. de la Plaine, 26190 Bouvante,...

 --- Résultats 3---
  Titre : Randonnée aquatique nature
  Lieu  : N/A — Bouvante
  Dates : 2025-05-09T09:00:00+00:00 → 2025-06-07T09:00:00+00:00
 Affichage dates : 9 mai - 7 juin 2025
  Texte : . Attention la sortie pourra être annulée ou reportée en cas de mauvaise météo ou s'il y a un nombre insuffisant de participants (moins de 4). Durée :...