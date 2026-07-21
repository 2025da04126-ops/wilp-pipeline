"""
Data Preparation Module for RecoMart Pipeline.
Handles data cleaning, preprocessing, encoding, normalization,
and exploratory data analysis.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    CLICKSTREAM_DIR, TRANSACTIONS_DIR, PRODUCTS_DIR,
    EXTERNAL_API_DIR, PROCESSED_DIR
)
from src.logger import get_logger

logger = get_logger("preparation")


class DataPreparation:
    """Handles data cleaning, preprocessing, and EDA."""
    
    def __init__(self):
        self.transactions = None
        self.products = None
        self.users = None
        self.clickstream = None
        self.external_data = None
        self.plots_dir = os.path.join(PROCESSED_DIR, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def load_raw_data(self):
        """Load all raw data sources."""
        logger.info("Loading raw data...")
        
        txn_path = os.path.join(TRANSACTIONS_DIR, 'transactions.csv')
        prod_path = os.path.join(PRODUCTS_DIR, 'products.csv')
        user_path = os.path.join(PRODUCTS_DIR, 'users.csv')
        click_path = os.path.join(CLICKSTREAM_DIR, 'clickstream.csv')
        ext_path = os.path.join(EXTERNAL_API_DIR, 'external_scores.csv')
        
        self.transactions = pd.read_csv(txn_path)
        self.products = pd.read_csv(prod_path)
        self.users = pd.read_csv(user_path)
        self.clickstream = pd.read_csv(click_path)
        self.external_data = pd.read_csv(ext_path)
        
        logger.info(f"Loaded - Transactions: {len(self.transactions)}, "
                   f"Products: {len(self.products)}, Users: {len(self.users)}, "
                   f"Clickstream: {len(self.clickstream)}")
    
    def clean_transactions(self) -> pd.DataFrame:
        """Clean transaction data: remove duplicates, handle missing values."""
        logger.info("Cleaning transactions data...")
        df = self.transactions.copy()
        
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicate rows")
        
        # Remove rows with missing critical fields
        critical_cols = ['transaction_id', 'user_id', 'product_id']
        df = df.dropna(subset=critical_cols)
        logger.info(f"Removed rows with missing critical fields. Remaining: {len(df)}")
        
        # Handle missing ratings - impute with median
        if df['rating'].isnull().sum() > 0:
            median_rating = df['rating'].median()
            df['rating'] = df['rating'].fillna(median_rating)
            logger.info(f"Imputed {df['rating'].isnull().sum()} missing ratings with median: {median_rating}")
        
        # Handle missing quantity - fill with 1
        df['quantity'] = df['quantity'].fillna(1)
        
        # Ensure rating is within valid range (1-5)
        df['rating'] = df['rating'].clip(1, 5)
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Add derived features
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        logger.info(f"Transaction cleaning complete. Records: {initial_count} -> {len(df)}")
        self.transactions = df
        return df
    
    def clean_clickstream(self) -> pd.DataFrame:
        """Clean clickstream data."""
        logger.info("Cleaning clickstream data...")
        df = self.clickstream.copy()
        
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Cap extreme page durations (> 30 minutes)
        df['page_duration_sec'] = df['page_duration_sec'].clip(0, 1800)
        
        # Add derived features
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        logger.info(f"Clickstream cleaning complete. Records: {initial_count} -> {len(df)}")
        self.clickstream = df
        return df
    
    def clean_products(self) -> pd.DataFrame:
        """Clean product data."""
        logger.info("Cleaning products data...")
        df = self.products.copy()
        
        # Remove duplicates by product_id
        df = df.drop_duplicates(subset=['product_id'])
        
        # Handle missing values
        df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].median())
        df['num_reviews'] = df['num_reviews'].fillna(0)
        
        # Normalize price (min-max scaling)
        df['price_normalized'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
        
        # Encode categories
        df['category_encoded'] = pd.Categorical(df['category']).codes
        
        logger.info(f"Product cleaning complete. Records: {len(df)}")
        self.products = df
        return df
    
    def clean_users(self) -> pd.DataFrame:
        """Clean user data."""
        logger.info("Cleaning user data...")
        df = self.users.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id'])
        
        # Encode gender
        gender_map = {'M': 0, 'F': 1, 'Other': 2}
        df['gender_encoded'] = df['gender'].map(gender_map)
        
        # Encode location
        df['location_encoded'] = pd.Categorical(df['location']).codes
        
        # Normalize age
        df['age_normalized'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        
        # Parse signup date
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days
        
        logger.info(f"User cleaning complete. Records: {len(df)}")
        self.users = df
        return df
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis and generate visualizations."""
        logger.info("Performing Exploratory Data Analysis...")
        
        # 1. User-Item Interaction Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Rating distribution
        if self.transactions is not None:
            self.transactions['rating'].hist(bins=5, ax=axes[0, 0], color='steelblue', edgecolor='black')
            axes[0, 0].set_title('Rating Distribution')
            axes[0, 0].set_xlabel('Rating')
            axes[0, 0].set_ylabel('Count')
        
        # Transactions per user
        if self.transactions is not None:
            user_txn_counts = self.transactions['user_id'].value_counts()
            user_txn_counts.hist(bins=30, ax=axes[0, 1], color='coral', edgecolor='black')
            axes[0, 1].set_title('Transactions per User')
            axes[0, 1].set_xlabel('Number of Transactions')
            axes[0, 1].set_ylabel('Number of Users')
        
        # Product popularity (top 20)
        if self.transactions is not None:
            top_products = self.transactions['product_id'].value_counts().head(20)
            top_products.plot(kind='bar', ax=axes[1, 0], color='seagreen')
            axes[1, 0].set_title('Top 20 Popular Products')
            axes[1, 0].set_xlabel('Product ID')
            axes[1, 0].set_ylabel('Transaction Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Category distribution
        if self.products is not None:
            self.products['category'].value_counts().plot(
                kind='pie', ax=axes[1, 1], autopct='%1.1f%%'
            )
            axes[1, 1].set_title('Product Category Distribution')
            axes[1, 1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'eda_overview.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Interaction sparsity matrix (sample)
        if self.transactions is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create user-item matrix (sample for visualization)
            sample_users = self.transactions['user_id'].value_counts().head(50).index
            sample_products = self.transactions['product_id'].value_counts().head(50).index
            
            sample_txn = self.transactions[
                self.transactions['user_id'].isin(sample_users) & 
                self.transactions['product_id'].isin(sample_products)
            ]
            
            interaction_matrix = sample_txn.pivot_table(
                index='user_id', columns='product_id', values='rating', aggfunc='mean'
            )
            
            sns.heatmap(interaction_matrix, cmap='YlOrRd', ax=ax, 
                       cbar_kws={'label': 'Rating'})
            ax.set_title('User-Item Interaction Heatmap (Sample)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'interaction_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Temporal patterns
        if self.transactions is not None and 'hour' in self.transactions.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            self.transactions['hour'].value_counts().sort_index().plot(
                kind='line', ax=axes[0], marker='o', color='purple'
            )
            axes[0].set_title('Transactions by Hour of Day')
            axes[0].set_xlabel('Hour')
            axes[0].set_ylabel('Count')
            
            self.transactions['day_of_week'].value_counts().sort_index().plot(
                kind='bar', ax=axes[1], color='teal'
            )
            axes[1].set_title('Transactions by Day of Week')
            axes[1].set_xlabel('Day (0=Mon, 6=Sun)')
            axes[1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'temporal_patterns.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Sparsity analysis
        if self.transactions is not None:
            n_users = self.transactions['user_id'].nunique()
            n_items = self.transactions['product_id'].nunique()
            n_interactions = len(self.transactions)
            sparsity = 1 - (n_interactions / (n_users * n_items))
            
            logger.info(f"Sparsity Analysis: {n_users} users, {n_items} items, "
                       f"{n_interactions} interactions, Sparsity: {sparsity:.4f}")
        
        logger.info(f"EDA plots saved to: {self.plots_dir}")
    
    def save_prepared_data(self):
        """Save cleaned and prepared data to processed directory."""
        logger.info("Saving prepared data...")
        
        if self.transactions is not None:
            self.transactions.to_csv(
                os.path.join(PROCESSED_DIR, 'transactions_clean.csv'), index=False
            )
        
        if self.products is not None:
            self.products.to_csv(
                os.path.join(PROCESSED_DIR, 'products_clean.csv'), index=False
            )
        
        if self.users is not None:
            self.users.to_csv(
                os.path.join(PROCESSED_DIR, 'users_clean.csv'), index=False
            )
        
        if self.clickstream is not None:
            self.clickstream.to_csv(
                os.path.join(PROCESSED_DIR, 'clickstream_clean.csv'), index=False
            )
        
        if self.external_data is not None:
            self.external_data.to_csv(
                os.path.join(PROCESSED_DIR, 'external_scores_clean.csv'), index=False
            )
        
        logger.info(f"Prepared data saved to: {PROCESSED_DIR}")
    
    def get_data_summary(self) -> Dict:
        """Generate a summary of prepared data."""
        summary = {
            'preparation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': {}
        }
        
        if self.transactions is not None:
            summary['datasets']['transactions'] = {
                'records': len(self.transactions),
                'unique_users': self.transactions['user_id'].nunique(),
                'unique_products': self.transactions['product_id'].nunique(),
                'avg_rating': round(self.transactions['rating'].mean(), 2),
                'date_range': f"{self.transactions['timestamp'].min()} to {self.transactions['timestamp'].max()}"
            }
        
        if self.products is not None:
            summary['datasets']['products'] = {
                'records': len(self.products),
                'categories': self.products['category'].nunique(),
                'avg_price': round(self.products['price'].mean(), 2)
            }
        
        if self.users is not None:
            summary['datasets']['users'] = {
                'records': len(self.users),
                'avg_age': round(self.users['age'].mean(), 1)
            }
        
        if self.clickstream is not None:
            summary['datasets']['clickstream'] = {
                'records': len(self.clickstream),
                'event_types': self.clickstream['event_type'].nunique()
            }
        
        return summary


def prepare_all_data() -> Dict:
    """Main function to run the complete data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Data Preparation Pipeline")
    logger.info("=" * 60)
    
    prep = DataPreparation()
    prep.load_raw_data()
    
    # Clean all datasets
    prep.clean_transactions()
    prep.clean_clickstream()
    prep.clean_products()
    prep.clean_users()
    
    # Perform EDA
    prep.perform_eda()
    
    # Save prepared data
    prep.save_prepared_data()
    
    # Get summary
    summary = prep.get_data_summary()
    
    logger.info("=" * 60)
    logger.info("Data Preparation Pipeline Complete")
    logger.info("=" * 60)
    
    return summary


if __name__ == "__main__":
    summary = prepare_all_data()
    print("\nData Preparation Summary:")
    for dataset, info in summary.get('datasets', {}).items():
        print(f"\n  {dataset}:")
        for key, value in info.items():
            print(f"    {key}: {value}")
