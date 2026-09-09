#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Centralisée - Détection de Fraude
"""

# ==================== NEO4J ====================
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # À MODIFIER avec votre mot de passe Neo4j

# ==================== DATASET ====================
DATASET_NAME = "ealaxi/paysim1"
DATASET_LIMIT_ROWS = None  # None = tous, ou 1000000 pour tester

# ==================== INGESTION ====================
INGEST_CHUNK_SIZE = 10000  # Nombre de lignes par batch
INGEST_SKIP_IF_EXISTS = True  # Sauter ingestion si données déjà présentes

# ==================== LOUVAIN ====================
LOUVAIN_GRAPH_LIMIT = 200000  # Transactions pour graphe Louvain
LOUVAIN_SHOW_TOP_N = 10  # Top N communautés à afficher

# ==================== GNN ====================
GNN_ENABLED = True  # Lancer le pipeline GNN ?
GNN_GRAPH_LIMIT = 500000  # Transactions pour entraînement GNN
GNN_EPOCHS = 100
GNN_HIDDEN_CHANNELS = 32
GNN_LEARNING_RATE = 0.01
GNN_WEIGHT_DECAY = 5e-4
GNN_DROPOUT = 0.5
GNN_BATCH_SIZE = 32  # Pour training (optionnel)
GNN_SHOW_TOP_N = 20  # Top N fraudeurs à afficher

# ==================== OUTPUT ====================
OUTPUT_DIR = "C:/FraudDetectionPipeline/outputs"
EXPORT_FORMAT = "json"  # json ou csv

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "C:/FraudDetectionPipeline/fraud_detection.log"

# ==================== PERFORMANCE ====================
USE_GPU = True  # Utiliser GPU si disponible (PyTorch)
WORKERS_DATALOADER = 4  # Workers pour DataLoader
RANDOM_SEED = 42  # Reproducibilité
