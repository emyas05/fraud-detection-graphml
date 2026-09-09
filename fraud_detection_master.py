#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT PRINCIPAL - Pipeline Complet de Détection de Fraude
===========================================================
Orchestre tous les modules : Ingestion → Louvain → GNN
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Imports locaux
try:
    from fraud_detection_pipeline import FraudDetectionPipeline
    from fraud_detection_gnn import FraudDetectionGNN
    import config
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("Assurez-vous que tous les fichiers sont dans le même répertoire")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Créer output dir
Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def print_header(title: str):
    """Affiche un en-tête formaté."""
    print("\n" + "="*70)
    print(f" {title}".center(70))
    print("="*70 + "\n")


def print_footer(message: str):
    """Affiche un pied de page."""
    print("\n" + "-"*70)
    print(f" {message}".center(70))
    print("-"*70 + "\n")


def main():
    """Exécute le pipeline complet."""
    
    start_time = datetime.now()
    results = {}
    
    print_header("🚀 PIPELINE COMPLET DÉTECTION DE FRAUDE")
    logger.info("="*70)
    logger.info("DÉMARRAGE DU PIPELINE COMPLET")
    logger.info("="*70)
    
    try:
        # ============================================================
        # PHASE 1 : INGESTION & STATISTIQUES
        # ============================================================
        print_header("PHASE 1️⃣  : INGESTION & STATISTIQUES")
        logger.info("Phase 1 : Ingestion et statistiques")
        
        pipeline = FraudDetectionPipeline(
            config.NEO4J_URI,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )
        
        try:
            pipeline.connect()
            
            # Vérifier si données existent
            if config.INGEST_SKIP_IF_EXISTS and pipeline.check_data_exists():
                logger.info("✓ Données déjà présentes, passage direct à l'analyse")
                print("✓ Données déjà chargées dans Neo4j")
            else:
                logger.info("Données non trouvées, démarrage de l'ingestion...")
                dataset_path = pipeline.download_dataset()
                csv_path = pipeline.find_csv_file(dataset_path)
                pipeline.create_indexes()
                pipeline.ingest_transactions(csv_path)
            
            # Statistiques
            pipeline.compute_statistics()
            results['ingestion'] = 'SUCCESS'
            
        finally:
            pipeline.close()
        
        print_footer("Phase 1 ✓ Complétée")
        
        # ============================================================
        # PHASE 2 : DÉTECTION DE COMMUNAUTÉS (LOUVAIN)
        # ============================================================
        print_header("PHASE 2️⃣  : DÉTECTION COMMUNAUTÉS (LOUVAIN)")
        logger.info("Phase 2 : Détection de communautés Louvain")
        
        pipeline = FraudDetectionPipeline(
            config.NEO4J_URI,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )
        
        try:
            pipeline.connect()
            communities = pipeline.detect_fraud_communities()
            results['louvain'] = {
                'status': 'SUCCESS',
                'communities_detected': len(communities),
                'top_suspicious': communities[:5]
            }
        finally:
            pipeline.close()
        
        print_footer("Phase 2 ✓ Complétée")
        
        # ============================================================
        # PHASE 3 : GNN (GRAPHSAGE)
        # ============================================================
        if config.GNN_ENABLED:
            print_header("PHASE 3️⃣  : GRAPH NEURAL NETWORK (GRAPHSAGE)")
            logger.info("Phase 3 : Entraînement GNN GraphSAGE")
            
            gnn_pipeline = FraudDetectionGNN(
                config.NEO4J_URI,
                config.NEO4J_USER,
                config.NEO4J_PASSWORD
            )
            
            # Entraîner
            train_success = gnn_pipeline.train_gnn(
                epochs=config.GNN_EPOCHS,
                limit=config.GNN_GRAPH_LIMIT
            )
            
            if train_success:
                # Prédire
                predictions = gnn_pipeline.predict_fraudsters()
                results['gnn'] = {
                    'status': 'SUCCESS',
                    'epochs': config.GNN_EPOCHS,
                    'top_fraudsters': predictions[:5]
                }
            else:
                results['gnn'] = {'status': 'FAILED'}
                logger.error("Erreur lors du GNN")
            
            print_footer("Phase 3 ✓ Complétée")
        else:
            logger.info("GNN désactivé (config.GNN_ENABLED=False)")
            results['gnn'] = {'status': 'SKIPPED'}
        
        # ============================================================
        # RAPPORT FINAL
        # ============================================================
        print_header("📊 RAPPORT FINAL")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'status': 'SUCCESS',
            'phases': results,
            'config': {
                'neo4j_uri': config.NEO4J_URI,
                'dataset': config.DATASET_NAME,
                'output_dir': config.OUTPUT_DIR
            }
        }
        
        # Exporter le rapport
        report_path = Path(config.OUTPUT_DIR) / "rapport_final.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"\n✓ Pipeline exécuté avec succès !")
        logger.info(f"  Durée totale : {duration:.2f}s")
        logger.info(f"  Rapport : {report_path}")
        
        print(f"""
        ╔═══════════════════════════════════════════════════════╗
        ║         ✓ PIPELINE COMPLÉTÉ AVEC SUCCÈS              ║
        ╠═══════════════════════════════════════════════════════╣
        ║                                                       ║
        ║  Phase 1 (Ingestion)    : ✓ Complétée               ║
        ║  Phase 2 (Louvain)      : ✓ Complétée               ║
        ║  Phase 3 (GNN)          : {'✓ Complétée' if config.GNN_ENABLED else '⊘ Désactivée'}               ║
        ║                                                       ║
        ║  Durée totale : {duration:.2f} secondes                   ║
        ║  Rapports    : {str(config.OUTPUT_DIR)}   ║
        ║                                                       ║
        ╚═══════════════════════════════════════════════════════╝
        """)
        
        # Afficher fichiers générés
        print("\n📁 Fichiers générés :")
        output_path = Path(config.OUTPUT_DIR)
        if output_path.exists():
            for file in output_path.glob("*.json"):
                size_kb = file.stat().st_size / 1024
                print(f"   ✓ {file.name:30s} ({size_kb:.1f} KB)")
        
        print("\n")
        
    except Exception as e:
        logger.error(f"❌ Erreur critique : {e}", exc_info=True)
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'status': 'FAILED',
            'error': str(e),
            'phases': results
        }
        
        report_path = Path(config.OUTPUT_DIR) / "rapport_final.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print("\n❌ ERREUR FATALE - Voir logs pour détails\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
