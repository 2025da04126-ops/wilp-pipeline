"""
Data Ingestion Module for RecoMart Pipeline.
Handles ingestion from CSV files and REST APIs with error handling,
retry mechanisms, and logging.
"""
import os
import sys
import time
import json
import shutil
import pandas as pd
import requests
from datetime import datetime
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    RAW_DIR, CLICKSTREAM_DIR, TRANSACTIONS_DIR, PRODUCTS_DIR,
    EXTERNAL_API_DIR, EXTERNAL_API_URL
)
from src.logger import get_logger

logger = get_logger("ingestion")


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


def retry_with_backoff(func, max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator-like function for retry logic with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                wait_time = backoff_factor ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                    raise DataIngestionError(
                        f"Failed after {max_retries} attempts: {str(e)}"
                    )
    return wrapper


def ingest_csv(source_path: str, destination_dir: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Ingest data from a CSV file source.
    
    Args:
        source_path: Path to the source CSV file
        destination_dir: Directory to store the ingested data
        filename: Name for the output file
    
    Returns:
        DataFrame of ingested data or None on failure
    """
    logger.info(f"Ingesting CSV data from: {source_path}")
    
    try:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Read the CSV
        df = pd.read_csv(source_path)
        
        # Add ingestion metadata
        df['_ingestion_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['_source_file'] = os.path.basename(source_path)
        
        # Save with timestamp partition
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(destination_dir, f"{filename}_{timestamp}.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(
            f"Successfully ingested {len(df)} records from {source_path} "
            f"-> {output_path}"
        )
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise DataIngestionError(str(e))
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {source_path}")
        raise DataIngestionError(f"Empty file: {source_path}")
    except Exception as e:
        logger.error(f"Unexpected error during CSV ingestion: {str(e)}")
        raise DataIngestionError(str(e))


def ingest_from_api(api_url: str, destination_dir: str, filename: str,
                    params: Optional[Dict] = None, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Ingest data from a REST API with retry mechanism.
    
    Args:
        api_url: URL of the API endpoint
        destination_dir: Directory to store the ingested data
        filename: Name for the output file
        params: Optional query parameters
        max_retries: Number of retry attempts
    
    Returns:
        DataFrame of ingested data or None on failure
    """
    logger.info(f"Ingesting data from API: {api_url}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise DataIngestionError("Unexpected API response format")
            
            # Add ingestion metadata
            df['_ingestion_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df['_source_api'] = api_url
            
            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(destination_dir, f"{filename}_{timestamp}.csv")
            df.to_csv(output_path, index=False)
            
            # Also save raw JSON response
            json_path = os.path.join(destination_dir, f"{filename}_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(
                f"Successfully ingested {len(df)} records from API -> {output_path}"
            )
            return df
            
        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: API timeout")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Connection error")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    logger.error(f"Failed to ingest from API after {max_retries} attempts")
    return None


def ingest_all_sources() -> Dict[str, Any]:
    """
    Main ingestion function that orchestrates data collection from all sources.
    
    Returns:
        Dictionary with ingestion results and metadata
    """
    logger.info("=" * 60)
    logger.info("Starting data ingestion pipeline...")
    logger.info("=" * 60)
    
    results = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sources': {},
        'status': 'success'
    }
    
    # 1. Ingest clickstream data (from local CSV)
    try:
        clickstream_source = os.path.join(CLICKSTREAM_DIR, 'clickstream.csv')
        if os.path.exists(clickstream_source):
            df = pd.read_csv(clickstream_source)
            results['sources']['clickstream'] = {
                'records': len(df),
                'status': 'success',
                'source_type': 'csv'
            }
            logger.info(f"Clickstream: {len(df)} records ingested")
        else:
            logger.warning("Clickstream source not found - generating synthetic data")
            from src.generate_data import generate_clickstream
            df = generate_clickstream()
            df.to_csv(clickstream_source, index=False)
            results['sources']['clickstream'] = {
                'records': len(df),
                'status': 'generated',
                'source_type': 'synthetic'
            }
    except Exception as e:
        logger.error(f"Clickstream ingestion failed: {str(e)}")
        results['sources']['clickstream'] = {'status': 'failed', 'error': str(e)}
    
    # 2. Ingest transaction data (from local CSV)
    try:
        txn_source = os.path.join(TRANSACTIONS_DIR, 'transactions.csv')
        if os.path.exists(txn_source):
            df = pd.read_csv(txn_source)
            results['sources']['transactions'] = {
                'records': len(df),
                'status': 'success',
                'source_type': 'csv'
            }
            logger.info(f"Transactions: {len(df)} records ingested")
        else:
            logger.warning("Transactions source not found - generating synthetic data")
            from src.generate_data import generate_transactions
            df = generate_transactions()
            df.to_csv(txn_source, index=False)
            results['sources']['transactions'] = {
                'records': len(df),
                'status': 'generated',
                'source_type': 'synthetic'
            }
    except Exception as e:
        logger.error(f"Transaction ingestion failed: {str(e)}")
        results['sources']['transactions'] = {'status': 'failed', 'error': str(e)}
    
    # 3. Ingest product data (from local CSV)
    try:
        prod_source = os.path.join(PRODUCTS_DIR, 'products.csv')
        if os.path.exists(prod_source):
            df = pd.read_csv(prod_source)
            results['sources']['products'] = {
                'records': len(df),
                'status': 'success',
                'source_type': 'csv'
            }
            logger.info(f"Products: {len(df)} records ingested")
        else:
            logger.warning("Products source not found - generating synthetic data")
            from src.generate_data import generate_products
            df = generate_products()
            df.to_csv(prod_source, index=False)
            results['sources']['products'] = {
                'records': len(df),
                'status': 'generated',
                'source_type': 'synthetic'
            }
    except Exception as e:
        logger.error(f"Product ingestion failed: {str(e)}")
        results['sources']['products'] = {'status': 'failed', 'error': str(e)}
    
    # 4. Ingest external API data
    try:
        api_source = os.path.join(EXTERNAL_API_DIR, 'external_scores.csv')
        if os.path.exists(api_source):
            df = pd.read_csv(api_source)
            results['sources']['external_api'] = {
                'records': len(df),
                'status': 'success',
                'source_type': 'api_cached'
            }
            logger.info(f"External API: {len(df)} records ingested")
        else:
            # Try to fetch from API, fall back to synthetic
            logger.info("Attempting external API ingestion...")
            df = ingest_from_api(EXTERNAL_API_URL, EXTERNAL_API_DIR, 'api_products')
            if df is None:
                logger.warning("API ingestion failed - generating synthetic data")
                from src.generate_data import generate_external_api_data
                df = generate_external_api_data()
                df.to_csv(api_source, index=False)
            results['sources']['external_api'] = {
                'records': len(df) if df is not None else 0,
                'status': 'success',
                'source_type': 'api'
            }
    except Exception as e:
        logger.error(f"External API ingestion failed: {str(e)}")
        results['sources']['external_api'] = {'status': 'failed', 'error': str(e)}
    
    results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save ingestion report
    report_path = os.path.join(RAW_DIR, 'ingestion_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"Data ingestion completed. Report saved to: {report_path}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    # First generate synthetic data if not present
    from src.generate_data import save_data
    save_data()
    
    # Then run the ingestion pipeline
    results = ingest_all_sources()
    print(json.dumps(results, indent=2))
