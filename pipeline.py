"""Module with prediction pipeline.
----------------------------------
To run with
default settings: run python pipeline.py
generate embeddings: run python pipeline.py --generate-embeddings.py
predict new drug-disease pairs: run python pipeline.py --predict-pairs new_pairs.csv
"""
#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from load_graph import BioKnowledgeGraph
from embedder import Node2VecEmbedder, save_embeddings, load_embeddings
from predictor import DrugDiseasePredictor

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Drug-Disease Prediction Pipeline")
    
    parser.add_argument("--nodes", type=str, default="data/Nodes.csv",
                       help="Path to nodes CSV file")
    parser.add_argument("--edges", type=str, default="data/Edges.csv",
                       help="Path to edges CSV file")
    parser.add_argument("--ground-truth", type=str, default="data/Ground_Truth.csv",
                       help="Path to ground truth CSV file")
    parser.add_argument("--embeddings-file", type=str, default="drug_disease_embeddings.pkl",
                       help="Path to save/load embeddings")
    parser.add_argument("--generate-embeddings", action="store_true",
                       help="Generate new embeddings instead of loading existing ones")
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Dimension of node embeddings")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data to use for testing")
    parser.add_argument("--predict-pairs", type=str,
                       help="Path to CSV file containing drug-disease pairs to predict")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set the logging level")
    
    return parser.parse_args()


def check_files_exist(files: Dict[str, str]) -> None:
    """Verify that required input files exist"""
    for name, filepath in files.items():
        if not Path(filepath).exists():
            raise FileNotFoundError(f"{name} file not found: {filepath}")


def save_results(results: Dict[str, Any], output_dir: str, args: argparse.Namespace) -> None:
    """Save results to files for later visualization."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    config = {
        'nodes_file': args.nodes,
        'edges_file': args.edges,
        'ground_truth_file': args.ground_truth,
        'embeddings_file': args.embeddings_file,
        'embedding_dim': args.embedding_dim,
        'test_size': args.test_size,
        'timestamp': timestamp
    }
    
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': ['ROC-AUC', 'Average Precision'],
        'train': [
            results['metrics']['train']['roc_auc'],
            results['metrics']['train']['avg_precision']
        ],
        'test': [
            results['metrics']['test']['roc_auc'],
            results['metrics']['test']['avg_precision']
        ]
    })
    metrics_df.to_csv(output_path / 'metrics.csv', index=False)
    
    # Save ROC curve data
    roc_data = {
        'train_fpr': results['metrics']['train']['fpr'],
        'train_tpr': results['metrics']['train']['tpr'],
        'test_fpr': results['metrics']['test']['fpr'],
        'test_tpr': results['metrics']['test']['tpr']
    }
    np.savez(output_path / 'roc_curves.npz', **roc_data)
    
    # Save feature importance
    feature_importance_df = pd.DataFrame(results['feature_importance'])
    feature_importance_df.to_csv(output_path / 'feature_importance.csv', index=False)
    
    # Save predictions if available
    if 'predictions' in results:
        pred_df = pd.DataFrame({
            'drug_id': [pair[0] for pair in results['predictions']['pairs']],
            'disease_id': [pair[1] for pair in results['predictions']['pairs']],
            'probability': results['predictions']['probabilities']
        })
        pred_df.to_csv(output_path / 'predictions.csv', index=False)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the main pipeline."""
    logger = logging.getLogger(__name__)
    
    # 1. Initialize knowledge graph
    logger.info("Initializing knowledge graph...")
    kg = BioKnowledgeGraph(args.nodes, args.edges)
    
    # 2. Handle embeddings
    logger.info("Setting up embedder...")
    embedder = Node2VecEmbedder(kg)
    
    if args.generate_embeddings:
        logger.info(f"Generating {args.embedding_dim}-dimensional embeddings (this may take ~15 minutes)...")
        embeddings = embedder.generate_embeddings(dimensions=args.embedding_dim)
        logger.info("Saving embeddings...")
        save_embeddings(embedder, args.embeddings_file)
    else:
        logger.info("Loading pre-computed embeddings...")
        load_embeddings(embedder, args.embeddings_file)
    
    # Verify embeddings loaded correctly
    emb_dim = len(next(iter(embedder.embeddings.values())))
    logger.info(f"Embedding dimension: {emb_dim}")
    
    # 3. Initialize and train predictor
    logger.info("Initializing predictor...")
    predictor = DrugDiseasePredictor(embedder)
    
    logger.info("Training model...")
    results = predictor.train_and_evaluate(
        ground_truth_file=args.ground_truth,
        test_size=args.test_size
    )
    
    # 4. Make predictions for new pairs if specified
    if args.predict_pairs:
        logger.info("Making predictions for new drug-disease pairs...")
        pairs_df = pd.read_csv(args.predict_pairs)
        new_pairs = list(zip(pairs_df['drug_id'], pairs_df['disease_id']))
        predictions = predictor.predict_new_pairs(new_pairs)
        
        results['predictions'] = {
            'pairs': new_pairs,
            'probabilities': predictions.tolist()
        }
    
    return results

def main() -> None:
    """Main entry point"""
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Check that required files exist
        required_files = {
            "Nodes": args.nodes,
            "Edges": args.edges,
            "Ground Truth": args.ground_truth
        }
        
        if not args.generate_embeddings:
            required_files["Embeddings"] = args.embeddings_file
            
        if args.predict_pairs:
            required_files["Prediction Pairs"] = args.predict_pairs
            
        check_files_exist(required_files)
        
        # Run the pipeline
        results = run_pipeline(args)
        
        # Save results
        save_results(results, args.output_dir, args)
        
        # Log final results
        logger.info("\nFinal Results:")
        logger.info(f"Train ROC-AUC: {results['metrics']['train']['roc_auc']:.3f}")
        logger.info(f"Test ROC-AUC: {results['metrics']['test']['roc_auc']:.3f}")
        logger.info(f"Train Avg Precision: {results['metrics']['train']['avg_precision']:.3f}")
        logger.info(f"Test Avg Precision: {results['metrics']['test']['avg_precision']:.3f}")
        
        # Log predictions if any
        if 'predictions' in results:
            logger.info("\nPredictions for new pairs:")
            for (drug, disease), prob in zip(
                results['predictions']['pairs'],
                results['predictions']['probabilities']
            ):
                logger.info(f"Pair: {drug} - {disease}")
                logger.info(f"Probability of treatment relationship: {prob:.3f}")
        
        logger.info(f"\nResults saved in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if args.log_level == "DEBUG":
            logger.exception("Detailed traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()