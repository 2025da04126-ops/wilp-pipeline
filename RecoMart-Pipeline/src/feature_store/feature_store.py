"""
Feature Store Module for RecoMart Pipeline.
Implements a simple feature store with versioning, metadata registry,
and retrieval capabilities for both training and inference.
"""
import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import FEATURES_DIR, BASE_DIR
from src.logger import get_logger

logger = get_logger("feature_store")

# Feature Store directory
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "data", "feature_store")
REGISTRY_FILE = os.path.join(FEATURE_STORE_DIR, "feature_registry.json")
VERSIONS_DIR = os.path.join(FEATURE_STORE_DIR, "versions")

# Ensure directories exist
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
os.makedirs(VERSIONS_DIR, exist_ok=True)


class FeatureRegistry:
    """Metadata registry for feature definitions and versions."""
    
    def __init__(self):
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry or create new one."""
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, 'r') as f:
                return json.load(f)
        return {'features': {}, 'versions': []}
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_feature(self, name: str, description: str, source: str,
                        transformation: str, dtype: str, entity: str):
        """Register a feature definition."""
        self.registry['features'][name] = {
            'name': name,
            'description': description,
            'source': source,
            'transformation': transformation,
            'dtype': dtype,
            'entity': entity,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self._save_registry()
        logger.info(f"Registered feature: {name}")
    
    def register_version(self, version_id: str, features: List[str],
                        description: str, data_hash: str):
        """Register a feature version."""
        version_entry = {
            'version_id': version_id,
            'features': features,
            'description': description,
            'data_hash': data_hash,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.registry['versions'].append(version_entry)
        self._save_registry()
        logger.info(f"Registered version: {version_id}")
    
    def get_feature_info(self, name: str) -> Optional[Dict]:
        """Get metadata for a specific feature."""
        return self.registry['features'].get(name)
    
    def list_features(self, entity: Optional[str] = None) -> List[Dict]:
        """List all registered features, optionally filtered by entity."""
        features = list(self.registry['features'].values())
        if entity:
            features = [f for f in features if f['entity'] == entity]
        return features
    
    def get_latest_version(self) -> Optional[Dict]:
        """Get the latest feature version."""
        if self.registry['versions']:
            return self.registry['versions'][-1]
        return None


class FeatureStore:
    """Simple feature store for managing features with versioning."""
    
    def __init__(self):
        self.registry = FeatureRegistry()
        self.user_features = None
        self.item_features = None
        self.interaction_features = None
    
    def ingest_features(self):
        """Load features from the features directory."""
        logger.info("Ingesting features into feature store...")
        
        user_path = os.path.join(FEATURES_DIR, 'user_features.csv')
        item_path = os.path.join(FEATURES_DIR, 'item_features.csv')
        interaction_path = os.path.join(FEATURES_DIR, 'interaction_features.csv')
        
        if os.path.exists(user_path):
            self.user_features = pd.read_csv(user_path)
            logger.info(f"Loaded {len(self.user_features)} user feature records")
        
        if os.path.exists(item_path):
            self.item_features = pd.read_csv(item_path)
            logger.info(f"Loaded {len(self.item_features)} item feature records")
        
        if os.path.exists(interaction_path):
            self.interaction_features = pd.read_csv(interaction_path)
            logger.info(f"Loaded {len(self.interaction_features)} interaction feature records")
    
    def register_all_features(self):
        """Register all feature definitions in the metadata registry."""
        logger.info("Registering feature definitions...")
        
        # User features
        user_feature_defs = [
            ("total_transactions", "Total number of purchases by user", "transactions", "count", "int", "user"),
            ("avg_rating_given", "Average rating given by user", "transactions", "mean(rating)", "float", "user"),
            ("unique_products_bought", "Number of unique products purchased", "transactions", "nunique(product_id)", "int", "user"),
            ("purchase_frequency", "Purchases per month", "transactions", "count/months", "float", "user"),
            ("engagement_score", "Composite engagement metric", "clickstream", "weighted_sum", "float", "user"),
            ("conversion_rate", "Purchase/view ratio", "transactions+clickstream", "ratio", "float", "user"),
            ("days_since_last_purchase", "Recency of last purchase", "transactions", "days_diff", "int", "user"),
        ]
        
        # Item features
        item_feature_defs = [
            ("total_purchases", "Total purchases of product", "transactions", "count", "int", "item"),
            ("avg_rating_received", "Average rating received", "transactions", "mean(rating)", "float", "item"),
            ("purchase_popularity", "Normalized purchase count", "transactions", "normalize", "float", "item"),
            ("view_to_purchase_ratio", "Cart adds / views", "clickstream", "ratio", "float", "item"),
            ("sentiment_score", "External sentiment analysis score", "external_api", "api_fetch", "float", "item"),
            ("popularity_score", "External popularity metric", "external_api", "api_fetch", "float", "item"),
            ("price_normalized", "Min-max normalized price", "products", "min_max_scale", "float", "item"),
        ]
        
        # Interaction features
        interaction_feature_defs = [
            ("avg_rating", "Average rating for user-item pair", "transactions", "mean", "float", "interaction"),
            ("num_interactions", "Number of interactions", "transactions", "count", "int", "interaction"),
            ("interaction_score", "Composite interaction metric", "multi-source", "weighted_sum", "float", "interaction"),
            ("view_count", "Number of views for user-item pair", "clickstream", "count", "int", "interaction"),
        ]
        
        all_defs = user_feature_defs + item_feature_defs + interaction_feature_defs
        for name, desc, source, transform, dtype, entity in all_defs:
            self.registry.register_feature(name, desc, source, transform, dtype, entity)
        
        logger.info(f"Registered {len(all_defs)} feature definitions")
    
    def create_version(self, description: str = "Feature snapshot") -> str:
        """Create a versioned snapshot of current features."""
        version_id = datetime.now().strftime('v%Y%m%d_%H%M%S')
        version_dir = os.path.join(VERSIONS_DIR, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        features_saved = []
        
        # Save each feature set
        if self.user_features is not None:
            path = os.path.join(version_dir, 'user_features.csv')
            self.user_features.to_csv(path, index=False)
            features_saved.append('user_features')
        
        if self.item_features is not None:
            path = os.path.join(version_dir, 'item_features.csv')
            self.item_features.to_csv(path, index=False)
            features_saved.append('item_features')
        
        if self.interaction_features is not None:
            path = os.path.join(version_dir, 'interaction_features.csv')
            self.interaction_features.to_csv(path, index=False)
            features_saved.append('interaction_features')
        
        # Compute data hash
        data_hash = self._compute_hash(version_dir)
        
        # Register version
        self.registry.register_version(version_id, features_saved, description, data_hash)
        
        logger.info(f"Created feature version: {version_id}")
        return version_id
    
    def get_features_for_training(self, version_id: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Retrieve features for model training."""
        logger.info(f"Retrieving features for training (version: {version_id or 'latest'})")
        
        if version_id:
            version_dir = os.path.join(VERSIONS_DIR, version_id)
        else:
            # Use latest version
            latest = self.registry.get_latest_version()
            if latest:
                version_dir = os.path.join(VERSIONS_DIR, latest['version_id'])
            else:
                # Use current features
                return {
                    'user_features': self.user_features,
                    'item_features': self.item_features,
                    'interaction_features': self.interaction_features
                }
        
        features = {}
        if os.path.exists(os.path.join(version_dir, 'user_features.csv')):
            features['user_features'] = pd.read_csv(os.path.join(version_dir, 'user_features.csv'))
        if os.path.exists(os.path.join(version_dir, 'item_features.csv')):
            features['item_features'] = pd.read_csv(os.path.join(version_dir, 'item_features.csv'))
        if os.path.exists(os.path.join(version_dir, 'interaction_features.csv')):
            features['interaction_features'] = pd.read_csv(os.path.join(version_dir, 'interaction_features.csv'))
        
        logger.info(f"Retrieved {len(features)} feature sets")
        return features
    
    def get_user_features(self, user_id: str) -> Optional[pd.Series]:
        """Retrieve features for a specific user (for inference)."""
        if self.user_features is not None:
            mask = self.user_features['user_id'] == user_id
            if mask.any():
                return self.user_features[mask].iloc[0]
        return None
    
    def get_item_features(self, product_id: str) -> Optional[pd.Series]:
        """Retrieve features for a specific item (for inference)."""
        if self.item_features is not None:
            mask = self.item_features['product_id'] == product_id
            if mask.any():
                return self.item_features[mask].iloc[0]
        return None
    
    def _compute_hash(self, directory: str) -> str:
        """Compute a hash of all files in a directory."""
        hasher = hashlib.md5()
        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
        return hasher.hexdigest()
    
    def get_store_summary(self) -> Dict:
        """Get a summary of the feature store."""
        return {
            'registered_features': len(self.registry.registry['features']),
            'versions': len(self.registry.registry['versions']),
            'latest_version': self.registry.get_latest_version(),
            'user_features_count': len(self.user_features) if self.user_features is not None else 0,
            'item_features_count': len(self.item_features) if self.item_features is not None else 0,
            'interaction_features_count': len(self.interaction_features) if self.interaction_features is not None else 0
        }


def run_feature_store() -> Dict:
    """Main function to initialize and populate the feature store."""
    logger.info("=" * 60)
    logger.info("Starting Feature Store Pipeline")
    logger.info("=" * 60)
    
    store = FeatureStore()
    
    # Ingest features
    store.ingest_features()
    
    # Register feature definitions
    store.register_all_features()
    
    # Create a versioned snapshot
    version_id = store.create_version("Initial feature snapshot")
    
    # Demonstrate retrieval
    features = store.get_features_for_training(version_id)
    logger.info(f"Retrieved {len(features)} feature sets for training")
    
    # Demonstrate single entity retrieval
    if store.user_features is not None and len(store.user_features) > 0:
        sample_user = store.user_features['user_id'].iloc[0]
        user_feats = store.get_user_features(sample_user)
        if user_feats is not None:
            logger.info(f"Sample user features for {sample_user}: "
                       f"{len(user_feats)} features retrieved")
    
    summary = store.get_store_summary()
    
    logger.info("=" * 60)
    logger.info("Feature Store Pipeline Complete")
    logger.info("=" * 60)
    
    return summary


if __name__ == "__main__":
    summary = run_feature_store()
    print("\nFeature Store Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
