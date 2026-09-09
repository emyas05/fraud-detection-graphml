#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline GraphML Avancé - GNN pour Détection de Fraude
=======================================================
Utilise PyTorch Geometric + GraphSAGE pour une détection intelligente
basée sur les patterns de réseau des transactions.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- 1. CHARGEUR NEO4J ---
class Neo4jGraphLoader:
    """Charge les données graphe depuis Neo4j pour entraînement GNN."""
    
    def __init__(self, uri: str, auth: tuple):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def fetch_graph_data(self, limit: int = 500000) -> pd.DataFrame:
        """
        Récupère les données du graphe pour l'entraînement.
        
        Args:
            limit: Nombre max de relations à charger
            
        Returns:
            DataFrame avec source, target, amount, label
        """
        logger.info(f"📥 Extraction de {limit:,} transactions depuis Neo4j...")
        
        query = f"""
        MATCH (src:Client)-[r:A_FAIT_TRANSACTION]->(dst:Client)
        RETURN 
            src.id as source, 
            dst.id as target, 
            r.amount as amount,
            r.isFraud as label,
            r.type as type
        LIMIT {limit}
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                df = pd.DataFrame([r.data() for r in result])
                
            logger.info(f"✓ {len(df):,} transactions chargées")
            logger.info(f"  Fraudes dans le sample : {(df['label'] == 1).sum():,}")
            return df
        except Neo4jError as e:
            logger.error(f"✗ Erreur Neo4j : {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()


# --- 2. CONSTRUCTEUR PYTORCH GEOMETRIC ---
def build_pytorch_data(df: pd.DataFrame) -> Data:
    """
    Transforme le DataFrame en objet PyTorch Geometric Data.
    
    Args:
        df: DataFrame avec source, target, amount, label
        
    Returns:
        Data object pour GNN
    """
    logger.info("⚙️ Construction du graphe PyTorch Geometric...")
    
    # 1. Mapping String IDs -> Indices entiers
    unique_nodes = pd.unique(df[['source', 'target']].values.ravel('K'))
    node_map = {id: i for i, id in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)
    logger.info(f"   Nœuds uniques : {num_nodes:,}")
    
    # 2. Construction Edge Index
    src = df['source'].map(node_map).values
    dst = df['target'].map(node_map).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 3. Features des Nœuds (Caractéristiques structurelles)
    logger.info("   Calcul des features structurelles...")
    node_features = {}
    
    # Degré entrant/sortant pour chaque nœud
    in_degree = np.zeros(num_nodes)
    out_degree = np.zeros(num_nodes)
    total_amount = np.zeros(num_nodes)
    fraud_count = np.zeros(num_nodes)
    
    for _, row in df.iterrows():
        src_idx = node_map[row['source']]
        dst_idx = node_map[row['target']]
        
        out_degree[src_idx] += 1
        in_degree[dst_idx] += 1
        total_amount[src_idx] += row['amount']
        if row['label'] == 1:
            fraud_count[src_idx] += 1
    
    # Normalisation
    scaler = StandardScaler()
    features = np.column_stack([
        in_degree,
        out_degree,
        scaler.fit_transform(total_amount.reshape(-1, 1)).ravel(),
        fraud_count
    ])
    
    x = torch.tensor(features, dtype=torch.float)
    logger.info(f"   Features shape : {x.shape}")
    
    # 4. Features des Arêtes (Montant normalisé)
    edge_attr = torch.tensor(df['amount'].values, dtype=torch.float).view(-1, 1)
    scaler_edge = StandardScaler()
    edge_attr = torch.tensor(
        scaler_edge.fit_transform(edge_attr), 
        dtype=torch.float
    )
    
    # 5. Labels (Y) - Nœuds fraudeurs
    fraudsters = set(df[df['label'] == 1]['source'].values)
    y = torch.zeros(num_nodes, dtype=torch.long)
    for node_id, idx in node_map.items():
        if node_id in fraudsters:
            y[idx] = 1
    
    logger.info(f"   Fraudeurs identifiés : {y.sum().item():,} ({100*y.sum().item()/num_nodes:.2f}%)")
    
    # 6. Création Data object
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        y=y,
        node_map=node_map  # Pour garder la trace
    )
    
    # 7. Masques train/test
    mask = torch.rand(num_nodes) < 0.8
    data.train_mask = mask
    data.test_mask = ~mask
    
    logger.info(f"✓ Data object prêt : {data}")
    return data


# --- 3. MODÈLE GNN GRAPHSAGE ---
class FraudGraphSAGE(torch.nn.Module):
    """
    GraphSAGE à 2 couches pour classification de nœuds (fraudeurs).
    GraphSAGE est excellent pour les grands graphes dynamiques.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout_prob = 0.5

    def forward(self, x, edge_index):
        # Couche 1
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        # Couche 2
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


# --- 4. PIPELINE COMPLET ---
class FraudDetectionGNN:
    """Pipeline GNN pour détection de fraude."""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data = None
        
        logger.info(f"🚀 Device : {self.device}")

    def train_gnn(self, epochs: int = 100, limit: int = 500000):
        """
        Entraîne le modèle GraphSAGE.
        
        Args:
            epochs: Nombre d'epochs
            limit: Nombre de transactions à charger
        """
        logger.info("\n" + "="*60)
        logger.info("ENTRAÎNEMENT DU MODÈLE GNN (GraphSAGE)")
        logger.info("="*60)
        
        # 1. Charger les données
        loader = Neo4jGraphLoader(
            self.neo4j_uri,
            (self.neo4j_user, self.neo4j_password)
        )
        try:
            df = loader.fetch_graph_data(limit=limit)
        finally:
            loader.close()
        
        if df.empty:
            logger.error("✗ Aucune donnée trouvée !")
            return False
        
        # 2. Construire le graphe PyTorch
        self.data = build_pytorch_data(df)
        
        # 3. Initialiser le modèle
        logger.info("\n🧠 Initialisation du modèle GraphSAGE...")
        self.model = FraudGraphSAGE(
            in_channels=self.data.x.size(1),
            hidden_channels=32,
            out_channels=2
        ).to(self.device)
        
        self.data = self.data.to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=5e-4
        )
        
        # 4. Boucle d'entraînement
        logger.info(f"\n📚 Entraînement pour {epochs} epochs...")
        
        self.model.train()
        best_acc = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            
            # Perte sur train set
            loss = F.nll_loss(
                out[self.data.train_mask],
                self.data.y[self.data.train_mask]
            )
            
            loss.backward()
            optimizer.step()
            
            # Évaluation
            if epoch % 10 == 0:
                self.model.eval()
                pred = self.model(self.data.x, self.data.edge_index).argmax(dim=1)
                
                train_acc = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).sum() / self.data.train_mask.sum()
                test_acc = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum() / self.data.test_mask.sum()
                
                logger.info(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    # Sauvegarder le meilleur modèle
                    torch.save(self.model.state_dict(), "C:/FraudDetectionPipeline/best_gnn_model.pth")
                
                self.model.train()
        
        logger.info(f"\n✓ Entraînement terminé. Best Test Acc: {best_acc:.4f}")
        return True

    def predict_fraudsters(self) -> dict:
        """
        Utilise le modèle entraîné pour prédire les fraudeurs.
        
        Returns:
            Dict avec scores de fraude par client
        """
        if self.model is None or self.data is None:
            logger.error("✗ Modèle non entraîné !")
            return {}
        
        logger.info("\n" + "="*60)
        logger.info("PRÉDICTIONS DE FRAUDE (GNN)")
        logger.info("="*60)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(out, dim=1)  # Probabilités
            preds = out.argmax(dim=1)
        
        # Extraire les scores de fraude
        fraud_scores = probs[:, 1].cpu().numpy()  # Prob de fraude
        
        # Reverse node map
        reverse_map = {v: k for k, v in self.data.node_map.items()}
        
        # Top fraudeurs
        top_fraudsters = []
        for node_idx, score in enumerate(fraud_scores):
            if node_idx in reverse_map:
                client_id = reverse_map[node_idx]
                top_fraudsters.append({
                    "client_id": client_id,
                    "fraud_score": float(score),
                    "prediction": int(preds[node_idx].item())
                })
        
        # Trier par score
        top_fraudsters.sort(key=lambda x: x["fraud_score"], reverse=True)
        
        logger.info(f"\nTop 20 clients à risque (GNN):")
        logger.info("-"*60)
        for idx, fraudster in enumerate(top_fraudsters[:20], 1):
            logger.info(f"{idx:2d}. Client {fraudster['client_id']:15s} | Score: {fraudster['fraud_score']:.4f}")
        
        # Exporter
        self._export_predictions(top_fraudsters[:100])
        
        return top_fraudsters

    def _export_predictions(self, predictions: list):
        """Exporte les prédictions en JSON."""
        try:
            export_path = Path("C:/FraudDetectionPipeline/gnn_fraud_scores.json")
            with open(export_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"\n💾 Prédictions exportées : {export_path}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur export : {e}")


def main():
    """Point d'entrée du pipeline GNN."""
    
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    logger.info("="*60)
    logger.info("PIPELINE GNN POUR DÉTECTION DE FRAUDE")
    logger.info("="*60 + "\n")
    
    # Pipeline
    gnn_pipeline = FraudDetectionGNN(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Entraîner
    if gnn_pipeline.train_gnn(epochs=100, limit=500000):
        # Prédire
        gnn_pipeline.predict_fraudsters()
        logger.info("\n✓ Pipeline GNN terminé !")
    else:
        logger.error("✗ Erreur lors de l'entraînement")
        sys.exit(1)


if __name__ == "__main__":
    main()
