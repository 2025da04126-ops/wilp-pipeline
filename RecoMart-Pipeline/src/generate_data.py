"""
Synthetic Data Generator for RecoMart Pipeline.
Generates realistic e-commerce data for testing the recommendation pipeline.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    NUM_USERS, NUM_PRODUCTS, NUM_TRANSACTIONS, NUM_CLICKSTREAM_EVENTS,
    CLICKSTREAM_DIR, TRANSACTIONS_DIR, PRODUCTS_DIR, EXTERNAL_API_DIR,
    RANDOM_STATE
)
from src.logger import get_logger

logger = get_logger("data_generator")


def generate_users(num_users: int = NUM_USERS) -> pd.DataFrame:
    """Generate synthetic user data."""
    np.random.seed(RANDOM_STATE)
    
    users = pd.DataFrame({
        'user_id': [f'U{str(i).zfill(4)}' for i in range(1, num_users + 1)],
        'age': np.random.randint(18, 65, num_users),
        'gender': np.random.choice(['M', 'F', 'Other'], num_users, p=[0.45, 0.45, 0.10]),
        'location': np.random.choice(
            ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
             'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow'],
            num_users
        ),
        'signup_date': [
            (datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 730))).strftime('%Y-%m-%d')
            for _ in range(num_users)
        ],
        'is_premium': np.random.choice([True, False], num_users, p=[0.2, 0.8])
    })
    return users


def generate_products(num_products: int = NUM_PRODUCTS) -> pd.DataFrame:
    """Generate synthetic product catalog."""
    np.random.seed(RANDOM_STATE + 1)
    
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen',
                  'Sports', 'Beauty', 'Toys', 'Food & Grocery']
    
    products = pd.DataFrame({
        'product_id': [f'P{str(i).zfill(4)}' for i in range(1, num_products + 1)],
        'product_name': [f'Product_{i}' for i in range(1, num_products + 1)],
        'category': np.random.choice(categories, num_products),
        'price': np.round(np.random.uniform(50, 5000, num_products), 2),
        'brand': np.random.choice(
            ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE',
             'BrandF', 'BrandG', 'BrandH'], num_products
        ),
        'avg_rating': np.round(np.random.uniform(1.0, 5.0, num_products), 1),
        'num_reviews': np.random.randint(0, 1000, num_products),
        'in_stock': np.random.choice([True, False], num_products, p=[0.85, 0.15])
    })
    return products


def generate_transactions(num_transactions: int = NUM_TRANSACTIONS,
                          num_users: int = NUM_USERS,
                          num_products: int = NUM_PRODUCTS) -> pd.DataFrame:
    """Generate synthetic transaction/purchase data."""
    np.random.seed(RANDOM_STATE + 2)
    
    transactions = pd.DataFrame({
        'transaction_id': [f'T{str(i).zfill(6)}' for i in range(1, num_transactions + 1)],
        'user_id': [f'U{str(np.random.randint(1, num_users + 1)).zfill(4)}' for _ in range(num_transactions)],
        'product_id': [f'P{str(np.random.randint(1, num_products + 1)).zfill(4)}' for _ in range(num_transactions)],
        'quantity': np.random.randint(1, 5, num_transactions),
        'rating': np.random.choice([1, 2, 3, 4, 5, None], num_transactions, 
                                   p=[0.05, 0.10, 0.20, 0.35, 0.25, 0.05]),
        'timestamp': [
            (datetime(2024, 1, 1) + timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )).strftime('%Y-%m-%d %H:%M:%S')
            for _ in range(num_transactions)
        ],
        'payment_method': np.random.choice(
            ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'COD'],
            num_transactions, p=[0.25, 0.20, 0.30, 0.15, 0.10]
        )
    })
    
    # Add some intentional data quality issues for validation testing
    # Add duplicate rows
    duplicates = transactions.sample(n=50, random_state=RANDOM_STATE)
    transactions = pd.concat([transactions, duplicates], ignore_index=True)
    
    # Add some missing values
    missing_idx = np.random.choice(transactions.index, size=100, replace=False)
    transactions.loc[missing_idx[:30], 'user_id'] = None
    transactions.loc[missing_idx[30:60], 'product_id'] = None
    transactions.loc[missing_idx[60:], 'quantity'] = None
    
    return transactions


def generate_clickstream(num_events: int = NUM_CLICKSTREAM_EVENTS,
                         num_users: int = NUM_USERS,
                         num_products: int = NUM_PRODUCTS) -> pd.DataFrame:
    """Generate synthetic clickstream/browsing data."""
    np.random.seed(RANDOM_STATE + 3)
    
    event_types = ['page_view', 'product_view', 'add_to_cart', 
                   'remove_from_cart', 'wishlist_add', 'search', 'purchase']
    event_probs = [0.30, 0.25, 0.15, 0.05, 0.08, 0.12, 0.05]
    
    clickstream = pd.DataFrame({
        'event_id': [f'E{str(i).zfill(7)}' for i in range(1, num_events + 1)],
        'user_id': [f'U{str(np.random.randint(1, num_users + 1)).zfill(4)}' for _ in range(num_events)],
        'product_id': [f'P{str(np.random.randint(1, num_products + 1)).zfill(4)}' for _ in range(num_events)],
        'event_type': np.random.choice(event_types, num_events, p=event_probs),
        'session_id': [f'S{str(np.random.randint(1, num_events // 5)).zfill(6)}' for _ in range(num_events)],
        'timestamp': [
            (datetime(2024, 1, 1) + timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )).strftime('%Y-%m-%d %H:%M:%S')
            for _ in range(num_events)
        ],
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], num_events, p=[0.55, 0.35, 0.10]),
        'page_duration_sec': np.random.exponential(30, num_events).astype(int)
    })
    return clickstream


def generate_external_api_data(num_products: int = NUM_PRODUCTS) -> pd.DataFrame:
    """Generate synthetic external API data (sentiment/popularity scores)."""
    np.random.seed(RANDOM_STATE + 4)
    
    api_data = pd.DataFrame({
        'product_id': [f'P{str(i).zfill(4)}' for i in range(1, num_products + 1)],
        'sentiment_score': np.round(np.random.uniform(-1.0, 1.0, num_products), 3),
        'popularity_score': np.round(np.random.uniform(0, 100, num_products), 2),
        'trending_rank': np.random.randint(1, 500, num_products),
        'social_mentions': np.random.randint(0, 5000, num_products),
        'competitor_price_ratio': np.round(np.random.uniform(0.7, 1.5, num_products), 2),
        'fetch_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    return api_data


def save_data():
    """Generate and save all synthetic data to raw directories."""
    logger.info("Starting synthetic data generation...")
    
    # Generate data
    users = generate_users()
    products = generate_products()
    transactions = generate_transactions()
    clickstream = generate_clickstream()
    external_data = generate_external_api_data()
    
    # Save to CSV
    users.to_csv(os.path.join(PRODUCTS_DIR, 'users.csv'), index=False)
    logger.info(f"Generated {len(users)} user records")
    
    products.to_csv(os.path.join(PRODUCTS_DIR, 'products.csv'), index=False)
    logger.info(f"Generated {len(products)} product records")
    
    transactions.to_csv(os.path.join(TRANSACTIONS_DIR, 'transactions.csv'), index=False)
    logger.info(f"Generated {len(transactions)} transaction records")
    
    clickstream.to_csv(os.path.join(CLICKSTREAM_DIR, 'clickstream.csv'), index=False)
    logger.info(f"Generated {len(clickstream)} clickstream events")
    
    external_data.to_csv(os.path.join(EXTERNAL_API_DIR, 'external_scores.csv'), index=False)
    logger.info(f"Generated {len(external_data)} external API records")
    
    logger.info("Synthetic data generation completed successfully!")
    
    return {
        'users': users,
        'products': products,
        'transactions': transactions,
        'clickstream': clickstream,
        'external_data': external_data
    }


if __name__ == "__main__":
    save_data()
