"""
Feature Engineering and Transformation Module for RecoMart Pipeline.
Creates features suitable for recommendation algorithms including
user activity frequency, average ratings, co-occurrence features, etc.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DIR, FEATURES_DIR
from src.logger import get_logger

logger = get_logger("transformation")


class FeatureEngineer:
    """Creates features for the recommendation system."""
    
    def __init__(self):
        self.transactions = None
        self.products = None
        self.users = None
        self.clickstream = None
        self.external_data = None
        self.user_features = None
        self.item_features = None
        self.interaction_features = None
    
    def load_prepared_data(self):
        """Load cleaned/prepared data."""
        logger.info("Loading prepared data for feature engineering...")
        
        self.transactions = pd.read_csv(os.path.join(PROCESSED_DIR, 'transactions_clean.csv'))
        self.products = pd.read_csv(os.path.join(PROCESSED_DIR, 'products_clean.csv'))
        self.users = pd.read_csv(os.path.join(PROCESSED_DIR, 'users_clean.csv'))
        self.clickstream = pd.read_csv(os.path.join(PROCESSED_DIR, 'clickstream_clean.csv'))
        self.external_data = pd.read_csv(os.path.join(PROCESSED_DIR, 'external_scores_clean.csv'))
        
        logger.info("Prepared data loaded successfully")
    
    def create_user_features(self) -> pd.DataFrame:
        """Create user-level features."""
        logger.info("Creating user features...")
        
        # Transaction-based features
        user_txn = self.transactions.groupby('user_id').agg(
            total_transactions=('transaction_id', 'count'),
            total_quantity=('quantity', 'sum'),
            avg_rating_given=('rating', 'mean'),
            std_rating_given=('rating', 'std'),
            unique_products_bought=('product_id', 'nunique'),
            avg_quantity_per_txn=('quantity', 'mean'),
            first_purchase=('timestamp', 'min'),
            last_purchase=('timestamp', 'max')
        ).reset_index()
        
        # Fill NaN std with 0 (users with single rating)
        user_txn['std_rating_given'] = user_txn['std_rating_given'].fillna(0)
        
        # Recency feature (days since last purchase)
        user_txn['last_purchase'] = pd.to_datetime(user_txn['last_purchase'])
        user_txn['days_since_last_purchase'] = (
            pd.Timestamp.now() - user_txn['last_purchase']
        ).dt.days
        
        # Purchase frequency (transactions per month)
        user_txn['first_purchase'] = pd.to_datetime(user_txn['first_purchase'])
        tenure_days = (user_txn['last_purchase'] - user_txn['first_purchase']).dt.days + 1
        user_txn['purchase_frequency'] = user_txn['total_transactions'] / (tenure_days / 30)
        
        # Clickstream-based features
        if self.clickstream is not None:
            user_clicks = self.clickstream.groupby('user_id').agg(
                total_events=('event_id', 'count'),
                total_sessions=('session_id', 'nunique'),
                avg_page_duration=('page_duration_sec', 'mean'),
                unique_products_viewed=('product_id', 'nunique'),
                cart_adds=('event_type', lambda x: (x == 'add_to_cart').sum()),
                searches=('event_type', lambda x: (x == 'search').sum())
            ).reset_index()
            
            # Engagement score
            user_clicks['engagement_score'] = (
                user_clicks['total_events'] * 0.3 +
                user_clicks['avg_page_duration'] * 0.3 +
                user_clicks['cart_adds'] * 0.2 +
                user_clicks['unique_products_viewed'] * 0.2
            )
            
            # Merge with transaction features
            user_txn = user_txn.merge(user_clicks, on='user_id', how='left')
        
        # Merge with user demographics
        user_features = user_txn.merge(
            self.users[['user_id', 'age', 'gender_encoded', 'location_encoded',
                       'is_premium', 'age_normalized', 'days_since_signup']],
            on='user_id', how='left'
        )
        
        # Conversion rate (purchases / views)
        if 'unique_products_viewed' in user_features.columns:
            user_features['conversion_rate'] = (
                user_features['unique_products_bought'] / 
                user_features['unique_products_viewed'].clip(lower=1)
            ).clip(upper=1.0)
        
        self.user_features = user_features
        logger.info(f"Created {len(user_features)} user feature records with "
                   f"{len(user_features.columns)} features")
        return user_features
    
    def create_item_features(self) -> pd.DataFrame:
        """Create item-level features."""
        logger.info("Creating item features...")
        
        # Transaction-based item features
        item_txn = self.transactions.groupby('product_id').agg(
            total_purchases=('transaction_id', 'count'),
            unique_buyers=('user_id', 'nunique'),
            avg_rating_received=('rating', 'mean'),
            std_rating_received=('rating', 'std'),
            total_quantity_sold=('quantity', 'sum'),
            avg_quantity_per_purchase=('quantity', 'mean')
        ).reset_index()
        
        item_txn['std_rating_received'] = item_txn['std_rating_received'].fillna(0)
        
        # Popularity score based on purchases
        max_purchases = item_txn['total_purchases'].max()
        item_txn['purchase_popularity'] = item_txn['total_purchases'] / max_purchases
        
        # Clickstream-based item features
        if self.clickstream is not None:
            item_clicks = self.clickstream.groupby('product_id').agg(
                total_views=('event_id', 'count'),
                unique_viewers=('user_id', 'nunique'),
                avg_view_duration=('page_duration_sec', 'mean'),
                cart_add_count=('event_type', lambda x: (x == 'add_to_cart').sum()),
                wishlist_count=('event_type', lambda x: (x == 'wishlist_add').sum())
            ).reset_index()
            
            # View-to-purchase conversion
            item_clicks['view_to_purchase_ratio'] = item_clicks['cart_add_count'] / item_clicks['total_views'].clip(lower=1)
            
            item_txn = item_txn.merge(item_clicks, on='product_id', how='left')
        
        # Merge with product metadata
        item_features = item_txn.merge(
            self.products[['product_id', 'category', 'price', 'brand',
                          'price_normalized', 'category_encoded', 'num_reviews', 'in_stock']],
            on='product_id', how='left'
        )
        
        # Merge with external data
        if self.external_data is not None:
            item_features = item_features.merge(
                self.external_data[['product_id', 'sentiment_score', 'popularity_score',
                                   'trending_rank', 'social_mentions']],
                on='product_id', how='left'
            )
        
        # Brand popularity
        brand_pop = item_features.groupby('brand')['total_purchases'].sum().reset_index()
        brand_pop.columns = ['brand', 'brand_total_purchases']
        item_features = item_features.merge(brand_pop, on='brand', how='left')
        
        self.item_features = item_features
        logger.info(f"Created {len(item_features)} item feature records with "
                   f"{len(item_features.columns)} features")
        return item_features
    
    def create_interaction_features(self) -> pd.DataFrame:
        """Create user-item interaction features for model training."""
        logger.info("Creating interaction features...")
        
        # Base interactions from transactions
        interactions = self.transactions[['user_id', 'product_id', 'rating', 'quantity', 'timestamp']].copy()
        
        # Aggregate multiple interactions between same user-item pair
        interaction_agg = interactions.groupby(['user_id', 'product_id']).agg(
            avg_rating=('rating', 'mean'),
            num_interactions=('rating', 'count'),
            total_quantity=('quantity', 'sum'),
            last_interaction=('timestamp', 'max')
        ).reset_index()
        
        # Add implicit feedback from clickstream
        if self.clickstream is not None:
            click_interactions = self.clickstream.groupby(['user_id', 'product_id']).agg(
                view_count=('event_id', 'count'),
                total_view_duration=('page_duration_sec', 'sum'),
                added_to_cart=('event_type', lambda x: int((x == 'add_to_cart').any())),
                added_to_wishlist=('event_type', lambda x: int((x == 'wishlist_add').any()))
            ).reset_index()
            
            interaction_agg = interaction_agg.merge(
                click_interactions, on=['user_id', 'product_id'], how='outer'
            )
        
        # Fill NaN values
        interaction_agg = interaction_agg.fillna(0)
        
        # Create a composite interaction score
        interaction_agg['interaction_score'] = (
            interaction_agg['avg_rating'] * 0.4 +
            interaction_agg['num_interactions'] * 0.2 +
            interaction_agg.get('view_count', 0) * 0.1 +
            interaction_agg.get('added_to_cart', 0) * 0.15 +
            interaction_agg.get('added_to_wishlist', 0) * 0.15
        )
        
        self.interaction_features = interaction_agg
        logger.info(f"Created {len(interaction_agg)} interaction feature records")
        return interaction_agg
    
    def create_user_item_matrix(self) -> Tuple[csr_matrix, list, list]:
        """Create sparse user-item interaction matrix for collaborative filtering."""
        logger.info("Creating user-item interaction matrix...")
        
        if self.interaction_features is None:
            self.create_interaction_features()
        
        # Map user and item IDs to indices
        user_ids = sorted(self.interaction_features['user_id'].unique())
        item_ids = sorted(self.interaction_features['product_id'].unique())
        
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        
        # Create sparse matrix
        rows = self.interaction_features['user_id'].map(user_to_idx)
        cols = self.interaction_features['product_id'].map(item_to_idx)
        values = self.interaction_features['avg_rating']
        
        # Filter out any NaN mappings
        valid_mask = rows.notna() & cols.notna()
        rows = rows[valid_mask].astype(int)
        cols = cols[valid_mask].astype(int)
        values = values[valid_mask]
        
        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(user_ids), len(item_ids))
        )
        
        sparsity = 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        logger.info(f"User-item matrix: {matrix.shape}, Sparsity: {sparsity:.4f}")
        
        return matrix, user_ids, item_ids
    
    def save_features(self):
        """Save all engineered features."""
        logger.info("Saving engineered features...")
        
        if self.user_features is not None:
            self.user_features.to_csv(
                os.path.join(FEATURES_DIR, 'user_features.csv'), index=False
            )
        
        if self.item_features is not None:
            self.item_features.to_csv(
                os.path.join(FEATURES_DIR, 'item_features.csv'), index=False
            )
        
        if self.interaction_features is not None:
            self.interaction_features.to_csv(
                os.path.join(FEATURES_DIR, 'interaction_features.csv'), index=False
            )
        
        # Save feature metadata
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_features': {
                'records': len(self.user_features) if self.user_features is not None else 0,
                'columns': list(self.user_features.columns) if self.user_features is not None else []
            },
            'item_features': {
                'records': len(self.item_features) if self.item_features is not None else 0,
                'columns': list(self.item_features.columns) if self.item_features is not None else []
            },
            'interaction_features': {
                'records': len(self.interaction_features) if self.interaction_features is not None else 0,
                'columns': list(self.interaction_features.columns) if self.interaction_features is not None else []
            }
        }
        
        import json
        with open(os.path.join(FEATURES_DIR, 'feature_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Features saved to: {FEATURES_DIR}")


# SQL Schema for feature storage
FEATURE_SCHEMA_SQL = """
-- SQL Schema for RecoMart Feature Store

CREATE TABLE IF NOT EXISTS user_features (
    user_id VARCHAR(10) PRIMARY KEY,
    total_transactions INT,
    total_quantity INT,
    avg_rating_given FLOAT,
    std_rating_given FLOAT,
    unique_products_bought INT,
    days_since_last_purchase INT,
    purchase_frequency FLOAT,
    total_events INT,
    engagement_score FLOAT,
    conversion_rate FLOAT,
    age INT,
    gender_encoded INT,
    is_premium BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS item_features (
    product_id VARCHAR(10) PRIMARY KEY,
    total_purchases INT,
    unique_buyers INT,
    avg_rating_received FLOAT,
    purchase_popularity FLOAT,
    total_views INT,
    view_to_purchase_ratio FLOAT,
    category VARCHAR(50),
    price FLOAT,
    price_normalized FLOAT,
    sentiment_score FLOAT,
    popularity_score FLOAT,
    trending_rank INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interaction_features (
    user_id VARCHAR(10),
    product_id VARCHAR(10),
    avg_rating FLOAT,
    num_interactions INT,
    total_quantity INT,
    view_count INT,
    interaction_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, product_id)
);

CREATE INDEX idx_interaction_user ON interaction_features(user_id);
CREATE INDEX idx_interaction_item ON interaction_features(product_id);
"""


def transform_all_data() -> Dict:
    """Main function to run the complete feature engineering pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    engineer = FeatureEngineer()
    engineer.load_prepared_data()
    
    # Create all feature sets
    user_features = engineer.create_user_features()
    item_features = engineer.create_item_features()
    interaction_features = engineer.create_interaction_features()
    
    # Create user-item matrix
    matrix, user_ids, item_ids = engineer.create_user_item_matrix()
    
    # Save features
    engineer.save_features()
    
    # Save SQL schema
    schema_path = os.path.join(FEATURES_DIR, 'feature_schema.sql')
    with open(schema_path, 'w') as f:
        f.write(FEATURE_SCHEMA_SQL)
    
    summary = {
        'user_features': len(user_features),
        'item_features': len(item_features),
        'interaction_features': len(interaction_features),
        'matrix_shape': matrix.shape,
        'matrix_sparsity': round(1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])), 4)
    }
    
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline Complete")
    logger.info("=" * 60)
    
    return summary


if __name__ == "__main__":
    summary = transform_all_data()
    print("\nFeature Engineering Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
