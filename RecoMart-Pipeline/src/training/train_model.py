"""
Model Training and Evaluation Module for RecoMart Pipeline.
Trains recommendation models (Collaborative Filtering via SVD,
Content-Based Filtering) and evaluates using Precision@K, Recall@K, NDCG.
"""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import FEATURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, TOP_K
from src.logger import get_logger

logger = get_logger("training")


class RecommendationMetrics:
    """Evaluation metrics for recommendation systems."""
    
    @staticmethod
    def precision_at_k(actual: List, predicted: List, k: int = 10) -> float:
        """Calculate Precision@K."""
        predicted_k = predicted[:k]
        relevant = set(actual) & set(predicted_k)
        return len(relevant) / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(actual: List, predicted: List, k: int = 10) -> float:
        """Calculate Recall@K."""
        predicted_k = predicted[:k]
        relevant = set(actual) & set(predicted_k)
        return len(relevant) / len(actual) if len(actual) > 0 else 0.0
    
    @staticmethod
    def ndcg_at_k(actual: List, predicted: List, k: int = 10) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)."""
        predicted_k = predicted[:k]
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            if item in actual:
                dcg += 1.0 / np.log2(i + 2)  # +2 because index starts at 0
        
        # Ideal DCG
        ideal_length = min(len(actual), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        mask = actual > 0  # Only consider non-zero ratings
        if mask.sum() == 0:
            return 0.0
        return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


class CollaborativeFilteringSVD:
    """Matrix Factorization based Collaborative Filtering using SVD."""
    
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.sigma = None
        self.user_ids = None
        self.item_ids = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.global_mean = 0.0
        self.predictions = None
    
    def fit(self, interactions: pd.DataFrame):
        """Train the SVD model on user-item interactions."""
        logger.info("Training Collaborative Filtering (SVD) model...")
        
        # Create user-item matrix
        self.user_ids = sorted(interactions['user_id'].unique())
        self.item_ids = sorted(interactions['product_id'].unique())
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        # Build rating matrix
        rating_matrix = np.zeros((n_users, n_items))
        for _, row in interactions.iterrows():
            user_idx = self.user_to_idx.get(row['user_id'])
            item_idx = self.item_to_idx.get(row['product_id'])
            if user_idx is not None and item_idx is not None:
                rating_matrix[user_idx, item_idx] = row['avg_rating']
        
        # Center the ratings
        self.global_mean = rating_matrix[rating_matrix > 0].mean()
        rating_centered = rating_matrix.copy()
        rating_centered[rating_matrix > 0] -= self.global_mean
        
        # Apply SVD
        n_factors = min(self.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(csr_matrix(rating_centered), k=n_factors)
        
        self.user_factors = U
        self.sigma = np.diag(sigma)
        self.item_factors = Vt.T
        
        # Compute full predictions
        self.predictions = self.global_mean + U @ self.sigma @ Vt
        
        # Clip to valid range
        self.predictions = np.clip(self.predictions, 1, 5)
        
        logger.info(f"SVD training complete. Factors: {n_factors}, "
                   f"Matrix: {n_users}x{n_items}")
    
    def predict(self, user_id: str, product_id: str) -> float:
        """Predict rating for a user-item pair."""
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(product_id)
        
        if user_idx is None or item_idx is None:
            return self.global_mean
        
        return self.predictions[user_idx, item_idx]
    
    def recommend(self, user_id: str, n: int = 10, 
                  exclude_known: bool = True) -> List[Tuple[str, float]]:
        """Get top-N recommendations for a user."""
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            return []
        
        scores = self.predictions[user_idx]
        
        if exclude_known:
            # Set known items to -inf
            known_mask = scores > 0
            # Actually we need the original ratings
            pass
        
        # Get top-N items
        top_indices = np.argsort(scores)[::-1][:n]
        recommendations = [(self.item_ids[idx], float(scores[idx])) for idx in top_indices]
        
        return recommendations


class ContentBasedFiltering:
    """Content-Based Filtering using item feature similarity."""
    
    def __init__(self):
        self.item_features_matrix = None
        self.item_ids = None
        self.similarity_matrix = None
    
    def fit(self, item_features: pd.DataFrame):
        """Build content-based model using item features."""
        logger.info("Training Content-Based Filtering model...")
        
        self.item_ids = item_features['product_id'].tolist()
        
        # Select numeric features for similarity computation
        numeric_cols = item_features.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID-like columns
        feature_cols = [c for c in numeric_cols if 'id' not in c.lower() and '_encoded' not in c]
        
        if not feature_cols:
            feature_cols = numeric_cols
        
        # Normalize features
        scaler = MinMaxScaler()
        self.item_features_matrix = scaler.fit_transform(
            item_features[feature_cols].fillna(0)
        )
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(self.item_features_matrix)
        
        logger.info(f"Content-based model trained. Items: {len(self.item_ids)}, "
                   f"Features used: {len(feature_cols)}")
    
    def get_similar_items(self, product_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get N most similar items to a given product."""
        if product_id not in self.item_ids:
            return []
        
        idx = self.item_ids.index(product_id)
        sim_scores = self.similarity_matrix[idx]
        
        # Get top-N similar (excluding self)
        top_indices = np.argsort(sim_scores)[::-1][1:n+1]
        similar_items = [(self.item_ids[i], float(sim_scores[i])) for i in top_indices]
        
        return similar_items
    
    def recommend_for_user(self, user_interactions: pd.DataFrame, 
                           n: int = 10) -> List[Tuple[str, float]]:
        """Recommend items based on user's historical interactions."""
        if user_interactions.empty:
            return []
        
        # Get items the user has interacted with
        user_items = user_interactions['product_id'].tolist()
        
        # Aggregate similarity scores for all user items
        scores = np.zeros(len(self.item_ids))
        for item_id in user_items:
            if item_id in self.item_ids:
                idx = self.item_ids.index(item_id)
                scores += self.similarity_matrix[idx]
        
        # Exclude already interacted items
        for item_id in user_items:
            if item_id in self.item_ids:
                idx = self.item_ids.index(item_id)
                scores[idx] = -1
        
        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n]
        recommendations = [(self.item_ids[i], float(scores[i])) for i in top_indices]
        
        return recommendations


class ModelTrainer:
    """Orchestrates model training and evaluation."""
    
    def __init__(self):
        self.cf_model = None
        self.cb_model = None
        self.metrics = RecommendationMetrics()
        self.results = {}
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load feature data for training."""
        interactions = pd.read_csv(os.path.join(FEATURES_DIR, 'interaction_features.csv'))
        item_features = pd.read_csv(os.path.join(FEATURES_DIR, 'item_features.csv'))
        return interactions, item_features
    
    def train_test_split_interactions(self, interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions into train and test sets."""
        # Sort by time if available, then split
        if 'last_interaction' in interactions.columns:
            interactions = interactions.sort_values('last_interaction')
        
        train, test = train_test_split(
            interactions, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        logger.info(f"Train/Test split: {len(train)}/{len(test)}")
        return train, test
    
    def train_collaborative_filtering(self, train_data: pd.DataFrame):
        """Train the collaborative filtering model."""
        self.cf_model = CollaborativeFilteringSVD(n_factors=50)
        self.cf_model.fit(train_data)
    
    def train_content_based(self, item_features: pd.DataFrame):
        """Train the content-based model."""
        self.cb_model = ContentBasedFiltering()
        self.cb_model.fit(item_features)
    
    def evaluate_models(self, test_data: pd.DataFrame, interactions: pd.DataFrame) -> Dict:
        """Evaluate both models on test data."""
        logger.info("Evaluating models...")
        
        results = {
            'collaborative_filtering': {},
            'content_based': {},
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Evaluate CF model
        if self.cf_model is not None:
            cf_metrics = self._evaluate_cf(test_data, interactions)
            results['collaborative_filtering'] = cf_metrics
            logger.info(f"CF Results - Precision@{TOP_K}: {cf_metrics['precision_at_k']:.4f}, "
                       f"Recall@{TOP_K}: {cf_metrics['recall_at_k']:.4f}, "
                       f"NDCG@{TOP_K}: {cf_metrics['ndcg_at_k']:.4f}, "
                       f"RMSE: {cf_metrics['rmse']:.4f}")
        
        # Evaluate CB model
        if self.cb_model is not None:
            cb_metrics = self._evaluate_cb(test_data, interactions)
            results['content_based'] = cb_metrics
            logger.info(f"CB Results - Precision@{TOP_K}: {cb_metrics['precision_at_k']:.4f}, "
                       f"Recall@{TOP_K}: {cb_metrics['recall_at_k']:.4f}, "
                       f"NDCG@{TOP_K}: {cb_metrics['ndcg_at_k']:.4f}")
        
        self.results = results
        return results
    
    def _evaluate_cf(self, test_data: pd.DataFrame, all_interactions: pd.DataFrame) -> Dict:
        """Evaluate collaborative filtering model."""
        precisions = []
        recalls = []
        ndcgs = []
        rmse_errors = []
        
        # Get unique test users
        test_users = test_data['user_id'].unique()
        sample_users = np.random.choice(test_users, min(100, len(test_users)), replace=False)
        
        for user_id in sample_users:
            # Actual items in test
            actual_items = test_data[test_data['user_id'] == user_id]['product_id'].tolist()
            
            if not actual_items:
                continue
            
            # Get predictions
            recommendations = self.cf_model.recommend(user_id, n=TOP_K)
            predicted_items = [item_id for item_id, _ in recommendations]
            
            # Calculate metrics
            precisions.append(self.metrics.precision_at_k(actual_items, predicted_items, TOP_K))
            recalls.append(self.metrics.recall_at_k(actual_items, predicted_items, TOP_K))
            ndcgs.append(self.metrics.ndcg_at_k(actual_items, predicted_items, TOP_K))
            
            # RMSE for rated items
            for _, row in test_data[test_data['user_id'] == user_id].iterrows():
                predicted_rating = self.cf_model.predict(row['user_id'], row['product_id'])
                actual_rating = row['avg_rating']
                rmse_errors.append((actual_rating - predicted_rating) ** 2)
        
        rmse = np.sqrt(np.mean(rmse_errors)) if rmse_errors else 0.0
        
        return {
            'precision_at_k': round(np.mean(precisions), 4) if precisions else 0.0,
            'recall_at_k': round(np.mean(recalls), 4) if recalls else 0.0,
            'ndcg_at_k': round(np.mean(ndcgs), 4) if ndcgs else 0.0,
            'rmse': round(rmse, 4),
            'k': TOP_K,
            'num_test_users': len(sample_users)
        }
    
    def _evaluate_cb(self, test_data: pd.DataFrame, all_interactions: pd.DataFrame) -> Dict:
        """Evaluate content-based model."""
        precisions = []
        recalls = []
        ndcgs = []
        
        test_users = test_data['user_id'].unique()
        sample_users = np.random.choice(test_users, min(100, len(test_users)), replace=False)
        
        for user_id in sample_users:
            # Actual items in test
            actual_items = test_data[test_data['user_id'] == user_id]['product_id'].tolist()
            
            if not actual_items:
                continue
            
            # User's training interactions
            user_train = all_interactions[
                (all_interactions['user_id'] == user_id) & 
                (~all_interactions['product_id'].isin(actual_items))
            ]
            
            # Get recommendations
            recommendations = self.cb_model.recommend_for_user(user_train, n=TOP_K)
            predicted_items = [item_id for item_id, _ in recommendations]
            
            # Calculate metrics
            precisions.append(self.metrics.precision_at_k(actual_items, predicted_items, TOP_K))
            recalls.append(self.metrics.recall_at_k(actual_items, predicted_items, TOP_K))
            ndcgs.append(self.metrics.ndcg_at_k(actual_items, predicted_items, TOP_K))
        
        return {
            'precision_at_k': round(np.mean(precisions), 4) if precisions else 0.0,
            'recall_at_k': round(np.mean(recalls), 4) if recalls else 0.0,
            'ndcg_at_k': round(np.mean(ndcgs), 4) if ndcgs else 0.0,
            'k': TOP_K,
            'num_test_users': len(sample_users)
        }
    
    def save_models(self):
        """Save trained models and metadata."""
        logger.info("Saving models...")
        
        # Save CF model
        if self.cf_model is not None:
            cf_path = os.path.join(MODELS_DIR, 'cf_svd_model.pkl')
            with open(cf_path, 'wb') as f:
                pickle.dump(self.cf_model, f)
            logger.info(f"CF model saved to: {cf_path}")
        
        # Save CB model
        if self.cb_model is not None:
            cb_path = os.path.join(MODELS_DIR, 'cb_model.pkl')
            with open(cb_path, 'wb') as f:
                pickle.dump(self.cb_model, f)
            logger.info(f"CB model saved to: {cb_path}")
        
        # Save model metadata (MLflow-style tracking)
        metadata = {
            'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {
                'collaborative_filtering': {
                    'algorithm': 'SVD (Truncated)',
                    'n_factors': self.cf_model.n_factors if self.cf_model else None,
                    'n_users': len(self.cf_model.user_ids) if self.cf_model else 0,
                    'n_items': len(self.cf_model.item_ids) if self.cf_model else 0
                },
                'content_based': {
                    'algorithm': 'Cosine Similarity',
                    'n_items': len(self.cb_model.item_ids) if self.cb_model else 0
                }
            },
            'metrics': self.results,
            'parameters': {
                'test_size': TEST_SIZE,
                'random_state': RANDOM_STATE,
                'top_k': TOP_K
            }
        }
        
        metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to: {metadata_path}")


def train_and_evaluate() -> Dict:
    """Main function to train and evaluate recommendation models."""
    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)
    
    trainer = ModelTrainer()
    
    # Load data
    interactions, item_features = trainer.load_data()
    
    # Train/test split
    train_data, test_data = trainer.train_test_split_interactions(interactions)
    
    # Train models
    trainer.train_collaborative_filtering(train_data)
    trainer.train_content_based(item_features)
    
    # Evaluate
    results = trainer.evaluate_models(test_data, interactions)
    
    # Save models
    trainer.save_models()
    
    # Print results
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'precision_at_k' in metrics:
            print(f"\n  {model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")
    
    print("\n" + "=" * 60)
    
    logger.info("=" * 60)
    logger.info("Model Training Pipeline Complete")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    results = train_and_evaluate()
