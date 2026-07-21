"""
Configuration settings for the RecoMart Pipeline.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Raw data subdirectories
CLICKSTREAM_DIR = os.path.join(RAW_DIR, "clickstream")
TRANSACTIONS_DIR = os.path.join(RAW_DIR, "transactions")
PRODUCTS_DIR = os.path.join(RAW_DIR, "products")
EXTERNAL_API_DIR = os.path.join(RAW_DIR, "external_api")

# Data generation parameters
NUM_USERS = 500
NUM_PRODUCTS = 200
NUM_TRANSACTIONS = 5000
NUM_CLICKSTREAM_EVENTS = 15000

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K = 10

# API simulation
EXTERNAL_API_URL = "https://fakestoreapi.com/products"

# Ensure directories exist
for dir_path in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR,
                 CLICKSTREAM_DIR, TRANSACTIONS_DIR, PRODUCTS_DIR, EXTERNAL_API_DIR]:
    os.makedirs(dir_path, exist_ok=True)
