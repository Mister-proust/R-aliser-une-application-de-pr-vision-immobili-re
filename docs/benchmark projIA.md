# RAPPORT DE BENCHMARK - PROJET IMMOBILIER IA

## À remplir par le binôme

---

## 1. Informations générales

| | |
|:---|:---|
| **Titre du projet** | Projet immobilier IA |
| **Binôme** | Étudiant 1 : Melody Duplaix Étudiant 2 : Maximilien Proust|
| **Date** | 9/3/26|
| **Version** | 1.0 |

---

## 2. Objectif du benchmark

*En quelques lignes, expliquez pourquoi vous réalisez ce benchmark et ce que vous cherchez à évaluer.*
L'objectif de ce benchmark est d'évaluer les différentes options technologiques disponibles pour la réalisation de notre projet d'agent immobilier intelligent. Nous cherchons à identifier les meilleures solutions en termes de frameworks d'orchestration IA, modèles de langage, bases de données, sources de données externes et interfaces utilisateur, afin de construire une application performante, fiable et adaptée aux besoins du marché immobilier.
> 

---

## 3. Benchmark des agents / chatbots immobiliers existants

*Analysez au moins 3 solutions existantes (sites web, applications, assistants) qui proposent des services similaires à votre projet.*

### 3.1 Solution 1 : Estimateur Immobilier

| Critère : Gratuit | Description : Estimateur ne nécessitant aucune inscription ni aucune information de contact.|
| **Nom / Lien** | https://www.estimateur-immobilier.com/|
| **Type** | ☐ Site web 
| **Fonctionnalités principales** | Estime le bien |
| **Sources de données utilisées** | historique des ventes effectuées dans votre secteur sur une période d’environ 5 ans|
| **Technologies présumées** | Statistiques |
| **Points forts** | gratuit, fourchettes de prix, sans contact|
| **Points faibles / limites** | Limité à Bordeaux sud visiblement même si fonctionne de façon national|
| **Ce que nous pourrions améliorer** | Y ajouter d'autres fonctionnalités que de la simple prédiction.  |

### 3.2 Solution 2 : PAP

| Critère : Gratuit | Description : Estimateur ne nécessitant aucune inscription ni aucune information de contact.|
| **Nom / Lien** | pap.fr |
| **Type** | ☐ Site web 
| **Fonctionnalités principales** | Estime le bien|
| **Sources de données utilisées** | DVF |
| **Technologies présumées** | Statistiques |
| **Points forts** | gratuit, fourchettes de prix, sans contact|
| **Points faibles / limites** | Estimation par fourchette |
| **Ce que nous pourrions améliorer** | Y ajouter d'autres fonctionnalités que de la simple prédiction.|

### 3.3 Solution 3 : Siana

| Critère : Application pour professionnel | Description : Application avec de multiple fonctionnalité |
| **Nom / Lien** | siana.app |
| **Type** | ☐ Application mobile |
| **Fonctionnalités principales** | Estimation de bien, recherche de biens, génération de prospect |
| **Sources de données utilisées** | Non indiqué |
| **Technologies présumées** | Probablement du RAG |
| **Points forts** | Beaucoup de fonctionnalités, application |
| **Points faibles / limites** | payant|
| **Ce que nous pourrions améliorer** | Le rendre gratuit et testable via le web |

*(Ajoutez autant de lignes que nécessaire)*

### 3.4 Synthèse du benchmark concurrentiel

*Quels sont les points communs entre ces solutions ? Quelles sont les lacunes du marché que votre projet pourrait combler ?*

Ils proposent tous une estimation. les lacunes du marché se portent autour de l'estimation immobilière qui est un marché volatile dépendant de l'offre et de la demande mais surtout de la qualité du bien qui n'est pas définissable en remplissant des simples formulaires. De plus, les données utilisées sont parfois datées de plusieurs mois/années et n'ont donc plus de réalité avec le marché actuel. 
Notre projet pourrait combler via une IA définissant la volatilité du marché. 


---

## 4. Benchmark des frameworks et outils d'orchestration IA

*Comparez les différentes options pour orchestrer votre agent IA.*

### 4.1 LangChain

| Critère | Évaluation |
|:---|:---|
| **Documentation** | ☑ Excellente ☐ Bonne ☐ Moyenne ☐ Insuffisante |
| **Facilité de prise en main** | ☐ Très facile ☐ Facile ☑ Complexe ☐ Très complexe |
| **Support des outils (tools)** | ☑ Natif ☐ Via extensions ☐ Limité ☐ Non supporté |
| **Communauté / Écosystème** | ☑ Très active ☐ Active ☐ Peu active ☐ Inexistante |
| **Intégration avec les LLM** | ☑ Nombreuses ☐ Quelques-unes ☐ Limitées ☐ Aucune |
| **Documentation en français** | ☐ Oui, abondante ☑ Quelques ressources ☐ Très peu ☐ Aucune |
| **Notre avis / commentaires** | Très complet pour un agent outillé (RAG, outils, mémoire, chaînes), mais API large donc plus de temps d'apprentissage. |

### 4.2 Autre framework 1 : LlamaIndex

| Critère | Évaluation |
|:---|:---|
| **Documentation** | ☐ Excellente ☑ Bonne ☐ Moyenne ☐ Insuffisante |
| **Facilité de prise en main** | ☐ Très facile ☑ Facile ☐ Complexe ☐ Très complexe |
| **Support des outils (tools)** | ☐ Natif ☑ Via extensions ☐ Limité ☐ Non supporté |
| **Communauté / Écosystème** | ☐ Très active ☑ Active ☐ Peu active ☐ Inexistante |
| **Intégration avec les LLM** | ☑ Nombreuses ☐ Quelques-unes ☐ Limitées ☐ Aucune |
| **Documentation en français** | ☐ Oui, abondante ☑ Quelques ressources ☐ Très peu ☐ Aucune |
| **Notre avis / commentaires** | Excellent pour les cas orientés RAG documentaire, un peu moins naturel pour une orchestration agent + outils métier complexes. |

### 4.3 Autre framework 2 : CrewAI

| Critère | Évaluation |
|:---|:---|
| **Documentation** | ☐ Excellente ☐ Bonne ☑ Moyenne ☐ Insuffisante |
| **Facilité de prise en main** | ☐ Très facile ☐ Facile ☑ Complexe ☐ Très complexe |
| **Support des outils (tools)** | ☑ Natif ☐ Via extensions ☐ Limité ☐ Non supporté |
| **Communauté / Écosystème** | ☐ Très active ☑ Active ☐ Peu active ☐ Inexistante |
| **Intégration avec les LLM** | ☐ Nombreuses ☑ Quelques-unes ☐ Limitées ☐ Aucune |
| **Documentation en français** | ☐ Oui, abondante ☐ Quelques ressources ☑ Très peu ☐ Aucune |
| **Notre avis / commentaires** | Pertinent pour des architectures multi-agents, mais surdimensionné pour un MVP en une semaine. |

### 4.4 Synthèse et choix motivé

*Quel framework avez-vous choisi et pourquoi ?*

**Framework retenu :** LangChain

**Justification :**
Nous retenons LangChain car il permet de construire rapidement un agent unique robuste avec outils (recherche, appel API, calcul, RAG) sans devoir développer l'orchestration from scratch. Son écosystème est mature, bien documenté et compatible avec la plupart des LLM du marché.

Pour notre contrainte de délai (1 semaine), c'est le meilleur compromis entre vitesse de prototypage et évolutivité : on peut livrer un MVP simple maintenant, puis enrichir progressivement (mémoire conversationnelle, traces, évaluation).




---

## 5. Benchmark des modèles de langage (LLM)

*Comparez les modèles que vous pourriez utiliser.*

| Critère | Option 1 : mistralai/Mistral-7B-Instruct-v0.2 | Option 2 : Mistral Large | Option 3 : Llama 3.1 70B Instruct |
|:---|:---|:---|:---|
| **Fournisseur** | Mistral AI | Meta (via hébergement tiers ou self-host) |
| **Type** | ☐ Open source ☑ Propriétaire | ☐ Open source ☑ Propriétaire | ☑ Open source ☐ Propriétaire |
| **Taille / Version** | Modèle compact optimisé coût/perf | Modèle généraliste haut de gamme | 70B paramètres |
| **Coût** | ☐ Gratuit ☑ Payant ☐ Freemium | ☐ Gratuit ☑ Payant ☐ Freemium | ☐ Gratuit ☐ Payant ☑ Freemium |
| **Performance / Qualité** | ☑ Excellente ☐ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne | ☐ Excellente ☐ Bonne ☑ Moyenne |
| **Vitesse d'inférence** | ☑ Rapide ☐ Moyenne ☐ Lente | ☑ Rapide ☐ Moyenne ☐ Lente | ☐ Rapide ☑ Moyenne ☐ Lente |
| **Support du français** | ☑ Excellent ☐ Bon ☐ Moyen ☐ Mauvais | ☑ Excellent ☐ Bon ☐ Moyen ☐ Mauvais | ☐ Excellent ☑ Bon ☐ Moyen ☐ Mauvais |
| **Facilité d'intégration** | ☑ Très facile ☐ Facile ☐ Complexe | ☐ Très facile ☑ Facile ☐ Complexe | ☐ Très facile ☐ Facile ☑ Complexe |
| **Limites (rate limits, etc.)** | Quotas API selon l'offre, coût à surveiller | Quotas API selon l'offre, coûts variables | Besoin d'infra si self-host, latence potentiellement plus élevée |

**Choix du modèle retenu :** mistralai/Mistral-7B-Instruct-v0.2

**Justification :**
Pour un MVP, mistralai/Mistral-7B-Instruct-v0.2 (souveraineté européenne et très bon français)  offre un bon équilibre qualité/vitesse/coût avec une intégration très simple dans Python et LangChain. Le support du français est solide, ce qui est essentiel pour les interactions utilisateurs et l'explication des estimations.




---

## 6. Benchmark des bases de données

*Comparez les options pour stocker vos données.*

| Critère | PostgreSQL | Autre option 1 : MongoDB | Autre option 2 : SQLite |
|:---|:---|:---|:---|
| **Type** | Relationnelle | Document | Relationnelle embarquée |
| **Support géospatial (PostGIS)** | ☑ Oui ☐ Non | ☑ Oui ☐ Non | ☐ Oui ☑ Non |
| **Performance** | ☑ Excellente ☐ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne | ☐ Excellente ☐ Bonne ☑ Moyenne |
| **Facilité d'installation** | ☐ Très facile ☑ Facile ☐ Complexe | ☐ Très facile ☑ Facile ☐ Complexe | ☑ Très facile ☐ Facile ☐ Complexe |
| **Documentation** | ☑ Excellente ☐ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne |
| **Communauté** | ☑ Très active ☐ Active ☐ Peu active | ☑ Très active ☐ Active ☐ Peu active | ☑ Très active ☐ Active ☐ Peu active |
| **Intégration avec Python** | ☑ Excellente ☐ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne |

**Choix de la base de données retenue :** PostgreSQL + PostGIS (et extension pgvector si besoin RAG)

**Justification :**
Le projet immobilier a besoin de données structurées, de filtres fiables et de géospatial (distance, zone, quartier). PostgreSQL + PostGIS couvre ces besoins nativement et reste un standard robuste en production.

Cette base est aussi très bien intégrée à Python/FastAPI et permet d'ajouter plus tard des capacités vectorielles via pgvector sans multiplier les briques techniques.




---

## 7. Benchmark des sources de données externes

*Évaluez les différentes sources de données que vous pourriez utiliser.*

### 7.1 API DVF (Demandes de Valeurs Foncières)

| Critère | Évaluation |
|:---|:---|
| **URL / Accès** | https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/ |
| **Type d'accès** | ☐ API REST ☑ Téléchargement de fichiers ☐ Base de données ☐ Autre : |
| **Format des données** | ☐ JSON ☑ CSV ☐ XML ☐ Autre : |
| **Données disponibles** | Transactions immobilières (prix, date, type de bien, surfaces, localisation administrative) |
| **Limitations (rate limit, volumétrie)** | Fichiers volumineux, nettoyage nécessaire, publication avec décalage temporel |
| **Qualité des données** | ☐ Excellente ☑ Bonne ☐ Moyenne ☐ Mauvaise |
| **Mise à jour / Fraîcheur** | Mise à jour semestrielle |
| **Documentation** | ☐ Excellente ☑ Bonne ☐ Moyenne ☐ Insuffisante |
| **Difficulté d'intégration** | ☐ Très facile ☑ Facile ☐ Complexe ☐ Très complexe |

### 7.2 Autre source 1 : Base Adresse Nationale (BAN)

| Critère | Évaluation |
|:---|:---|
| **URL / Accès** | https://api-adresse.data.gouv.fr/ |
| **Type d'accès** | ☑ API REST ☐ Téléchargement de fichiers ☐ Base de données ☐ Autre : |
| **Format des données** | ☑ JSON ☐ CSV ☐ XML ☐ Autre : |
| **Données disponibles** | Géocodage/reverse géocodage, normalisation d'adresse, coordonnées géographiques |
| **Limitations (rate limit, volumétrie)** | Requêtes massives à encadrer, mise en cache recommandée |
| **Qualité des données** | ☑ Excellente ☐ Bonne ☐ Moyenne ☐ Mauvaise |
| **Mise à jour / Fraîcheur** | Fréquente (base nationale maintenue en continu) |
| **Documentation** | ☑ Excellente ☐ Bonne ☐ Moyenne ☐ Insuffisante |
| **Difficulté d'intégration** | ☑ Très facile ☐ Facile ☐ Complexe ☐ Très complexe |

### 7.3 Autre source 2 : INSEE (données territoriales et socio-économiques)

| Critère | Évaluation |
|:---|:---|
| **URL / Accès** | https://api.insee.fr/ |
| **Type d'accès** | ☑ API REST ☐ Téléchargement de fichiers ☐ Base de données ☐ Autre : |
| **Format des données** | ☑ JSON ☐ CSV ☐ XML ☐ Autre : |
| **Données disponibles** | Démographie, revenus, emploi, indicateurs de territoire (commune, IRIS, département) |
| **Limitations (rate limit, volumétrie)** | Nécessite une clé, nomenclatures à maîtriser, granularité variable selon indicateurs |
| **Qualité des données** | ☑ Excellente ☐ Bonne ☐ Moyenne ☐ Mauvaise |
| **Mise à jour / Fraîcheur** | Selon indicateur (annuel/trimestriel) |
| **Documentation** | ☐ Excellente ☑ Bonne ☐ Moyenne ☐ Insuffisante |
| **Difficulté d'intégration** | ☐ Très facile ☐ Facile ☑ Complexe ☐ Très complexe |

### 7.4 Synthèse des sources retenues

| Source | Données utilisées | Méthode d'intégration | Priorité |
|:---|:---|:---|:---|
| DVF | Historique de ventes et prix/m² par zone | ETL batch (ingestion CSV vers PostgreSQL) | ☑ Haute ☐ Moyenne ☐ Basse |
| BAN | Géocodage des adresses et coordonnées | Appel API REST + cache local | ☑ Haute ☐ Moyenne ☐ Basse |
| INSEE | Contexte socio-économique local | Appel API REST planifié + stockage agrégé | ☐ Haute ☑ Moyenne ☐ Basse |

---

## 8. Benchmark des interfaces utilisateur / frontend

*Comparez les options pour créer l'interface de votre application.*

| Critère | Streamlit | Gradio | Autre : FastAPI + React | Autre : Flask + Jinja |
|:---|:---|:---|:---|:---|
| **Rapidité de développement** | ☑ Excellente ☐ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne | ☐ Excellente ☐ Bonne ☑ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne |
| **Composants chat intégrés** | ☑ Oui ☐ Non | ☑ Oui ☐ Non | ☐ Oui ☑ Non | ☐ Oui ☑ Non |
| **Personnalisation** | ☐ Excellente ☐ Bonne ☑ Moyenne | ☐ Excellente ☐ Bonne ☑ Moyenne | ☑ Excellente ☐ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne |
| **Documentation** | ☐ Excellente ☑ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne | ☑ Excellente ☐ Bonne ☐ Moyenne | ☐ Excellente ☑ Bonne ☐ Moyenne |
| **Communauté** | ☑ Très active ☐ Active ☐ Peu active | ☐ Très active ☑ Active ☐ Peu active | ☑ Très active ☐ Active ☐ Peu active | ☐ Très active ☑ Active ☐ Peu active |
| **Facilité de déploiement** | ☑ Très facile ☐ Facile ☐ Complexe | ☐ Très facile ☑ Facile ☐ Complexe | ☐ Très facile ☐ Facile ☑ Complexe | ☐ Très facile ☑ Facile ☐ Complexe |

**Choix de l'interface retenue :** Streamlit ou FastApi/jinja selon l'avancée

**Justification :**
Streamlit est le plus adapté à notre contrainte de temps : on peut livrer vite une interface claire avec chat, filtres et visualisations sans surcoût frontend important.

Pour une V2 orientée produit, une migration vers FastAPI + React restera possible afin d'augmenter la personnalisation et la scalabilité UI.




---

## 9. Synthèse générale du benchmark

### 9.1 Récapitulatif des choix technologiques

| Domaine | Technologie choisie |
|:---|:---|
| Framework d'orchestration IA | LangChain |
| Modèle de langage (LLM) | mistralai/Mistral-7B-Instruct-v0.2 |
| Base de données | PostgreSQL + PostGIS (+ pgvector si nécessaire) |
| Backend / API | FastAPI |
| Frontend / Interface | Streamlit |
| Sources de données externes | DVF + BAN + INSEE |

### 9.2 Justification globale

*Expliquez en quelques paragraphes pourquoi votre stack technologique est la plus adaptée au projet, compte tenu des contraintes (délai d'une semaine, compétences, objectifs).*

Notre priorité est de produire en une semaine un MVP crédible, démontrable, et techniquement propre. La combinaison LangChain + mistralai/Mistral-7B-Instruct-v0.2 permet d'obtenir rapidement un agent conversationnel performant en français, capable d'utiliser des outils métiers et de formuler des réponses explicables à l'utilisateur.

Le socle de données PostgreSQL + PostGIS est un choix stratégique pour l'immobilier : il répond aux besoins de requêtes structurées, de filtrage et d'analyse spatiale. Les sources DVF, BAN et INSEE apportent respectivement l'historique de transactions, la qualité de géolocalisation et le contexte territorial, ce qui renforce la pertinence des estimations.

Enfin, Streamlit et FastAPI offrent une trajectoire pragmatique : itération rapide en phase projet, puis possibilité d'industrialiser progressivement (API plus complète, observabilité, tests, optimisation coût/latence du LLM).


### 9.3 Enseignements clés du benchmark

*Qu'avez-vous appris en réalisant ce benchmark ? Quelles bonnes pratiques allez-vous adopter ?*

Nous avons appris qu'un bon agent immobilier IA ne dépend pas seulement du LLM, mais surtout de la qualité des données, de la fraîcheur des sources et de la capacité à justifier les réponses. Le benchmark montre qu'une stack simple et maîtrisée vaut mieux qu'une architecture trop ambitieuse au départ.

Bonnes pratiques retenues :
- Commencer par un périmètre MVP clair (estimation + explication + sources citées).
- Mettre en place un pipeline de données reproductible (ETL DVF, normalisation adresses).
- Tracer les prompts/réponses et mesurer la qualité (tests manuels + cas de référence).
- Séparer le socle API/data de l'interface pour faciliter l'évolution vers une V2.
- Surveiller coût, latence et hallucinations avec des garde-fous (validation, seuils, fallback).


---

## 10. Annexes

*Liens vers les ressources consultées, articles, documentations, etc.*

- LangChain docs : https://python.langchain.com/
- LlamaIndex docs : https://docs.llamaindex.ai/
- CrewAI docs : https://docs.crewai.com/
- OpenAI API docs : https://platform.openai.com/docs
- Mistral docs : https://docs.mistral.ai/
- PostgreSQL : https://www.postgresql.org/docs/
- PostGIS : https://postgis.net/documentation/
- DVF (data.gouv) : https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/
- API Adresse (BAN) : https://api-adresse.data.gouv.fr/
- API INSEE : https://api.insee.fr/
- Streamlit docs : https://docs.streamlit.io/
- FastAPI docs : https://fastapi.tiangolo.com/


