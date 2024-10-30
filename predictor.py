import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from typing import Dict, List, Tuple, Optional
from embedder import Node2VecEmbedder

class DrugDiseasePredictor:
    def __init__(self, embedder: Node2VecEmbedder):
        """
        Initialize predictor with trained embeddings
        
        Parameters:
        -----------
        embedder: Node2VecEmbedder
            Trained node2vec embedder with generated embeddings
        """
        self.embedder = embedder
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def _create_pair_features(self, drug_id: str, disease_id: str) -> Optional[np.ndarray]:
        """
        Create features for a drug-disease pair
        
        Parameters:
        -----------
        drug_id: str
            ID of the drug node
        disease_id: str
            ID of the disease node
            
        Returns:
        --------
        Optional[np.ndarray]
            Feature vector for the pair, None if embeddings not found
        """
        try:
            # Get embeddings
            drug_emb = self.embedder.get_node_embedding(drug_id)
            disease_emb = self.embedder.get_node_embedding(disease_id)
            
            if drug_emb is None or disease_emb is None:
                return None
                
            # Create features
            features = [
                drug_emb * disease_emb,        # Hadamard product
                np.abs(drug_emb - disease_emb), # L1 distance
                (drug_emb + disease_emb) / 2,   # Average
                np.maximum(drug_emb, disease_emb) # Element-wise maximum
            ]
            
            return np.concatenate(features)
            
        except Exception as e:
            print(f"Error creating features for {drug_id}-{disease_id}: {str(e)}")
            return None
    
    def prepare_data_from_ground_truth(self, 
                                     ground_truth_file: str,
                                     test_size: float = 0.2) -> Dict:
        """
        Prepare training data from ground truth file
        
        Parameters:
        -----------
        ground_truth_file: str
            Path to ground truth CSV file
        test_size: float
            Proportion of data to use for testing
            
        Returns:
        --------
        Dict
            Dictionary containing train/test splits and metadata
        """
        # Load ground truth data
        gt_df = pd.read_csv(ground_truth_file)
        print(f"Loaded {len(gt_df)} ground truth pairs")
        
        # Create features for all pairs
        X = []
        y = []
        valid_pairs = []
        
        for idx, row in gt_df.iterrows():
            features = self._create_pair_features(row['source'], row['target'])
            
            if features is not None:
                X.append(features)
                y.append(row['y'])
                valid_pairs.append((row['source'], row['target']))
            
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx} pairs...")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nCreated features for {len(X)} pairs")
        print(f"Positive examples: {sum(y)}")
        print(f"Negative examples: {len(y) - sum(y)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(
            X_scaled, y, valid_pairs, test_size=test_size, 
            random_state=42, stratify=y
        )
        
        return {
            'train': {
                'X': X_train,
                'y': y_train,
                'pairs': pairs_train
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'pairs': pairs_test
            },
            'metadata': {
                'num_features': X.shape[1],
                'num_positive': sum(y),
                'num_negative': len(y) - sum(y)
            }
        }
    
    def train_and_evaluate(self, 
                          ground_truth_file: str,
                          test_size: float = 0.2) -> Dict:
        """
        Train model and evaluate performance
        
        Parameters:
        -----------
        ground_truth_file: str
            Path to ground truth CSV file
        test_size: float
            Proportion of data to use for testing
            
        Returns:
        --------
        Dict
            Dictionary containing evaluation metrics and model
        """
        # Prepare data
        data = self.prepare_data_from_ground_truth(ground_truth_file, test_size)
        
        # Train model
        # TODO: potentially fine-tune parameters with gridsearch cv
        # TODO: model selection

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )
        
        print("\nTraining model...")
        self.model.fit(data['train']['X'], data['train']['y'])
        
        # Make predictions
        train_probs = self.model.predict_proba(data['train']['X'])[:, 1]
        test_probs = self.model.predict_proba(data['test']['X'])[:, 1]
        
        # Calculate ROC curves
        train_fpr, train_tpr, _ = roc_curve(data['train']['y'], train_probs)
        test_fpr, test_tpr, _ = roc_curve(data['test']['y'], test_probs)
        
        # Calculate metrics
        metrics = {
            'train': {
                'roc_auc': roc_auc_score(data['train']['y'], train_probs),
                'avg_precision': average_precision_score(data['train']['y'], train_probs),
                'fpr': train_fpr.tolist(),  # Convert to list for JSON serialization
                'tpr': train_tpr.tolist(),
                'predictions': train_probs.tolist(),
                'actual': data['train']['y'].tolist()
            },
            'test': {
                'roc_auc': roc_auc_score(data['test']['y'], test_probs),
                'avg_precision': average_precision_score(data['test']['y'], test_probs),
                'fpr': test_fpr.tolist(),
                'tpr': test_tpr.tolist(),
                'predictions': test_probs.tolist(),
                'actual': data['test']['y'].tolist()
            }
        }
        
        # Get feature importance
        # Create feature names based on the operations we did in _create_pair_features
        emb_dim = len(next(iter(self.embedder.embeddings.values())))
        operations = ['hadamard', 'l1_distance', 'average', 'max']
        feature_names = []
        for op in operations:
            for i in range(emb_dim):
                feature_names.append(f'{op}_dim_{i}')
        
        # Create feature importance dictionary
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_,
            'operation': [op for op in operations for _ in range(emb_dim)]
        }).sort_values('importance', ascending=False)
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'data': data
        }
    
    def predict_new_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Predict probabilities for new drug-disease pairs
        
        Parameters:
        -----------
        pairs: List[Tuple[str, str]]
            List of (drug_id, disease_id) pairs
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Create features for new pairs
        X = []
        valid_pairs = []
        
        for drug_id, disease_id in pairs:
            features = self._create_pair_features(drug_id, disease_id)
            if features is not None:
                X.append(features)
                valid_pairs.append((drug_id, disease_id))
        
        if not X:
            return np.array([])
            
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities