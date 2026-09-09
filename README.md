Détection de Fraude Basée sur Graphes avec Neo4j et Graph ML

📋 Objectif du Projet

Développer un pipeline complet et robuste pour détecter les fraudes dans un réseau de transactions financières (dataset PaySim 6M+) en utilisant :

Neo4j : Base de données graphe pour la modélisation des transactions
Graph ML : Algorithmes avancés (Louvain, GraphSAGE, PageRank)
PyTorch Geometric : Deep Learning sur graphes
🎯 Architecture du Projet

FraudDetectionPipeline/
├── fraud_detection_pipeline.py      # Ingestion + stats + Louvain
├── fraud_detection_gnn.py           # GNN GraphSAGE
├── fraud_detection_master.py        # Script principal (LANCER CECI)
├── README.md                        # Ce fichier
├── requirements.txt                 # Dépendances
├── config.py                        # Configuration
└── outputs/
    ├── fraud_communities.json       # Résultats Louvain
    ├── gnn_fraud_scores.json        # Prédictions GNN
    └── rapport_final.json           # Rapport complet
🚀 Installation & Démarrage

1. Prérequis

Python 3.10+
Neo4j Desktop (lancé et configuré)
GPU optionnel (CUDA pour accélération)
2. Installation des dépendances

pip install -r requirements.txt
3. Configuration Neo4j

Modifier config.py :

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Votre mot de passe
4. Lancer le pipeline complet

python fraud_detection_master.py
📊 Phases du Projet

Phase 1 : Ingestion & Statistiques

✅ Télécharge dataset PaySim (6M lignes, 0.46 GB)
✅ Charge dans Neo4j par chunks (optimisation RAM)
✅ Crée index pour performance
✅ Génère statistiques globales
Output : 9M+ nœuds Client, 6.3M+ transactions

Phase 2 : Détection de Communautés (Louvain)

✅ Algorithme Louvain (Graph Community Detection)
✅ Identifie communautés frauduleuses
✅ Calcul modularité
✅ Top 10 communautés suspectes
Output : fraud_communities.json

Phase 3 : Graph Neural Networks (GraphSAGE)

✅ Construit graphe PyTorch Geometric
✅ Features structurelles (degré, montants, fraudes)
✅ Entraîne GraphSAGE (2 couches)
✅ Prédit scores de fraude par client
Output : gnn_fraud_scores.json, modèle sauvegardé

📈 Résultats Attendus

STATISTIQUES:
├── Clients (nœuds)              : 9,073,900
├── Transactions (relations)     : 6,362,620
├── Taux de fraude global        : 0.13%
└── Montant total                : $1.144 Trillion

COMMUNAUTÉS SUSPECTES (Louvain):
├── Communauté #5 : 12.45% fraude
├── Communauté #12: 8.67% fraude
└── ...

GNN PREDICTIONS:
├── Top fraudeur 1 : Score 0.98
├── Top fraudeur 2 : Score 0.95
└── ...
🔧 Techniques Graph ML Utilisées

Technique	Utilisation
Louvain	Détection de communautés frauduleuses
GraphSAGE	Classification des nœuds (fraudeurs)
PageRank	Centralité des acteurs (optionnel)
Node2Vec	Embeddings pour clustering (optionnel)
Network Analysis	Propriétés structurelles du graphe
📁 Fichiers Principaux

fraud_detection_pipeline.py

Pipeline d'ingestion et analyse Louvain

Classe FraudDetectionPipeline
Ingestion par chunks (10k lignes)
Détection de communautés Louvain
Export JSON
fraud_detection_gnn.py

Pipeline GNN avec GraphSAGE

Classe Neo4jGraphLoader
Classe FraudGraphSAGE (torch.nn.Module)
Classe FraudDetectionGNN
Entraînement + prédictions
fraud_detection_master.py

Script orchestrateur principal

Exécute tous les pipelines séquentiellement
Génère rapport final
Gère les erreurs
💾 Données

Dataset : PaySim1 (Kaggle)

Taille : 0.46 GB, 6.3M lignes
Colonnes : nameOrig, nameDest, amount, type, step, isFraud
Fraudes : 8,213 (0.13%)
Source : https://kaggle.com/ealaxi/paysim1
Schéma Neo4j

(:Client {id: "C1"})-[:A_FAIT_TRANSACTION {amount, type, step, isFraud}]->(:Client {id: "C2"})
🎓 Livrables du Projet

 Pipeline robuste d'ingestion (6M+ lignes)
 Modélisation graphe Neo4j
 Algorithme Louvain (détection communautés)
 GNN GraphSAGE entraîné
 Prédictions de fraude
 Rapports et statistiques
 Documentation complète
📞 Support

Pour des questions sur :

Neo4j : Voir documentation officielle (cypher, indexes)
Graph ML : Tutorials PyTorch Geometric
Pipeline : Consulter les logs et résultats JSON
✅ Checklist Finale

 Neo4j Desktop lancé
 config.py configuré (URI, auth)
 Dépendances installées (pip install -r requirements.txt)
 Premier lancement : python fraud_detection_master.py
 Vérifier outputs JSON générés
 Analyser résultats et communautés suspectes

 ## Team
[Imane Ait said +hatim maachi +ziad essadki]
Imane Ait Said — Master's Student, Data Science & Intelligent Systems, FST Fez
