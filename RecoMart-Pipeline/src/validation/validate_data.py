"""
Data Validation Module for RecoMart Pipeline.
Performs data quality checks including missing values, duplicates,
schema validation, range checks, and generates a quality report.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    RAW_DIR, CLICKSTREAM_DIR, TRANSACTIONS_DIR, PRODUCTS_DIR,
    EXTERNAL_API_DIR, PROCESSED_DIR
)
from src.logger import get_logger

logger = get_logger("validation")


# Schema definitions for each data source
SCHEMAS = {
    'transactions': {
        'required_columns': ['transaction_id', 'user_id', 'product_id', 'quantity', 'rating', 'timestamp'],
        'dtypes': {
            'transaction_id': 'str',
            'user_id': 'str',
            'product_id': 'str',
            'quantity': 'numeric',
            'rating': 'numeric',
            'timestamp': 'datetime'
        },
        'ranges': {
            'rating': (1, 5),
            'quantity': (1, 100)
        }
    },
    'products': {
        'required_columns': ['product_id', 'product_name', 'category', 'price', 'brand'],
        'dtypes': {
            'product_id': 'str',
            'product_name': 'str',
            'category': 'str',
            'price': 'numeric',
            'brand': 'str'
        },
        'ranges': {
            'price': (0.01, 100000),
            'avg_rating': (1, 5)
        }
    },
    'clickstream': {
        'required_columns': ['event_id', 'user_id', 'product_id', 'event_type', 'timestamp'],
        'dtypes': {
            'event_id': 'str',
            'user_id': 'str',
            'product_id': 'str',
            'event_type': 'str',
            'timestamp': 'datetime'
        },
        'valid_values': {
            'event_type': ['page_view', 'product_view', 'add_to_cart',
                          'remove_from_cart', 'wishlist_add', 'search', 'purchase'],
            'device': ['mobile', 'desktop', 'tablet']
        }
    },
    'external_scores': {
        'required_columns': ['product_id', 'sentiment_score', 'popularity_score'],
        'dtypes': {
            'product_id': 'str',
            'sentiment_score': 'numeric',
            'popularity_score': 'numeric'
        },
        'ranges': {
            'sentiment_score': (-1, 1),
            'popularity_score': (0, 100)
        }
    }
}


class DataValidator:
    """Validates data quality and generates quality reports."""
    
    def __init__(self, df: pd.DataFrame, source_name: str):
        self.df = df
        self.source_name = source_name
        self.schema = SCHEMAS.get(source_name, {})
        self.issues: List[Dict] = []
        self.metrics: Dict[str, Any] = {}
    
    def validate_schema(self) -> bool:
        """Check if required columns are present."""
        required = self.schema.get('required_columns', [])
        missing_cols = [col for col in required if col not in self.df.columns]
        
        if missing_cols:
            self.issues.append({
                'type': 'schema_mismatch',
                'severity': 'critical',
                'details': f"Missing columns: {missing_cols}"
            })
            logger.error(f"[{self.source_name}] Missing columns: {missing_cols}")
            return False
        
        logger.info(f"[{self.source_name}] Schema validation passed")
        return True
    
    def check_missing_values(self) -> Dict[str, float]:
        """Check for missing values in each column."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        missing_report = {}
        for col in self.df.columns:
            if missing[col] > 0:
                missing_report[col] = {
                    'count': int(missing[col]),
                    'percentage': float(missing_pct[col])
                }
                severity = 'warning' if missing_pct[col] < 5 else 'critical'
                self.issues.append({
                    'type': 'missing_values',
                    'severity': severity,
                    'column': col,
                    'details': f"{missing[col]} missing ({missing_pct[col]}%)"
                })
        
        self.metrics['missing_values'] = missing_report
        total_missing = missing.sum()
        logger.info(f"[{self.source_name}] Total missing values: {total_missing}")
        return missing_report
    
    def check_duplicates(self) -> Dict[str, int]:
        """Check for duplicate records."""
        total_dupes = self.df.duplicated().sum()
        
        # Check by primary key if available
        id_col = None
        for col in ['transaction_id', 'event_id', 'product_id']:
            if col in self.df.columns:
                id_col = col
                break
        
        pk_dupes = 0
        if id_col:
            pk_dupes = self.df[id_col].duplicated().sum()
        
        dupe_report = {
            'total_duplicates': int(total_dupes),
            'primary_key_duplicates': int(pk_dupes)
        }
        
        if total_dupes > 0:
            self.issues.append({
                'type': 'duplicates',
                'severity': 'warning',
                'details': f"{total_dupes} duplicate rows found"
            })
        
        self.metrics['duplicates'] = dupe_report
        logger.info(f"[{self.source_name}] Duplicates: {total_dupes} total, {pk_dupes} by PK")
        return dupe_report
    
    def check_ranges(self) -> Dict[str, Dict]:
        """Validate numerical columns are within expected ranges."""
        ranges = self.schema.get('ranges', {})
        range_report = {}
        
        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns:
                numeric_col = pd.to_numeric(self.df[col], errors='coerce')
                below = (numeric_col < min_val).sum()
                above = (numeric_col > max_val).sum()
                
                range_report[col] = {
                    'min_expected': min_val,
                    'max_expected': max_val,
                    'actual_min': float(numeric_col.min()) if not numeric_col.isna().all() else None,
                    'actual_max': float(numeric_col.max()) if not numeric_col.isna().all() else None,
                    'below_range': int(below),
                    'above_range': int(above)
                }
                
                if below > 0 or above > 0:
                    self.issues.append({
                        'type': 'range_violation',
                        'severity': 'warning',
                        'column': col,
                        'details': f"{below} below min, {above} above max"
                    })
        
        self.metrics['range_checks'] = range_report
        return range_report
    
    def check_valid_values(self) -> Dict[str, Dict]:
        """Check categorical columns have valid values."""
        valid_values = self.schema.get('valid_values', {})
        validity_report = {}
        
        for col, allowed in valid_values.items():
            if col in self.df.columns:
                invalid = self.df[~self.df[col].isin(allowed) & self.df[col].notna()]
                validity_report[col] = {
                    'allowed_values': allowed,
                    'invalid_count': len(invalid),
                    'invalid_samples': invalid[col].unique()[:5].tolist() if len(invalid) > 0 else []
                }
                
                if len(invalid) > 0:
                    self.issues.append({
                        'type': 'invalid_values',
                        'severity': 'warning',
                        'column': col,
                        'details': f"{len(invalid)} rows with invalid values"
                    })
        
        self.metrics['valid_values'] = validity_report
        return validity_report
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute basic statistics about the dataset."""
        stats = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'column_types': self.df.dtypes.apply(str).to_dict()
        }
        
        # Numeric column statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = self.df[numeric_cols].describe().to_dict()
        
        self.metrics['statistics'] = stats
        return stats
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all validation checks and generate a complete report."""
        logger.info(f"Running validation for: {self.source_name}")
        logger.info("-" * 40)
        
        self.validate_schema()
        self.check_missing_values()
        self.check_duplicates()
        self.check_ranges()
        self.check_valid_values()
        self.compute_statistics()
        
        report = {
            'source': self.source_name,
            'validation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_issues': len(self.issues),
            'critical_issues': len([i for i in self.issues if i['severity'] == 'critical']),
            'warning_issues': len([i for i in self.issues if i['severity'] == 'warning']),
            'issues': self.issues,
            'metrics': self.metrics,
            'overall_quality_score': self._compute_quality_score()
        }
        
        logger.info(f"[{self.source_name}] Quality Score: {report['overall_quality_score']}/100")
        logger.info(f"[{self.source_name}] Issues: {report['total_issues']} "
                   f"({report['critical_issues']} critical, {report['warning_issues']} warnings)")
        
        return report
    
    def _compute_quality_score(self) -> float:
        """Compute an overall data quality score (0-100)."""
        score = 100.0
        
        # Deduct for missing values
        if 'missing_values' in self.metrics:
            total_missing_pct = sum(
                v['percentage'] for v in self.metrics['missing_values'].values()
            ) / max(len(self.df.columns), 1)
            score -= min(total_missing_pct * 2, 30)
        
        # Deduct for duplicates
        if 'duplicates' in self.metrics:
            dupe_pct = (self.metrics['duplicates']['total_duplicates'] / max(len(self.df), 1)) * 100
            score -= min(dupe_pct, 20)
        
        # Deduct for range violations
        if 'range_checks' in self.metrics:
            for col_report in self.metrics['range_checks'].values():
                violations = col_report.get('below_range', 0) + col_report.get('above_range', 0)
                violation_pct = (violations / max(len(self.df), 1)) * 100
                score -= min(violation_pct, 10)
        
        # Deduct for schema issues
        critical_count = len([i for i in self.issues if i['severity'] == 'critical'])
        score -= critical_count * 15
        
        return round(max(score, 0), 1)


def validate_all_sources() -> Dict[str, Any]:
    """Validate all data sources and generate a comprehensive quality report."""
    logger.info("=" * 60)
    logger.info("Starting Data Validation Pipeline")
    logger.info("=" * 60)
    
    all_reports = {}
    
    # Validate Transactions
    txn_path = os.path.join(TRANSACTIONS_DIR, 'transactions.csv')
    if os.path.exists(txn_path):
        df = pd.read_csv(txn_path)
        validator = DataValidator(df, 'transactions')
        all_reports['transactions'] = validator.run_all_checks()
    
    # Validate Products
    prod_path = os.path.join(PRODUCTS_DIR, 'products.csv')
    if os.path.exists(prod_path):
        df = pd.read_csv(prod_path)
        validator = DataValidator(df, 'products')
        all_reports['products'] = validator.run_all_checks()
    
    # Validate Clickstream
    click_path = os.path.join(CLICKSTREAM_DIR, 'clickstream.csv')
    if os.path.exists(click_path):
        df = pd.read_csv(click_path)
        validator = DataValidator(df, 'clickstream')
        all_reports['clickstream'] = validator.run_all_checks()
    
    # Validate External API Data
    ext_path = os.path.join(EXTERNAL_API_DIR, 'external_scores.csv')
    if os.path.exists(ext_path):
        df = pd.read_csv(ext_path)
        validator = DataValidator(df, 'external_scores')
        all_reports['external_scores'] = validator.run_all_checks()
    
    # Save comprehensive report
    report_path = os.path.join(PROCESSED_DIR, 'data_quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info(f"Validation complete. Report saved to: {report_path}")
    logger.info("=" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT SUMMARY")
    print("=" * 60)
    for source, report in all_reports.items():
        print(f"\n  {source.upper()}")
        print(f"    Quality Score: {report['overall_quality_score']}/100")
        print(f"    Total Issues: {report['total_issues']}")
        print(f"    Critical: {report['critical_issues']} | Warnings: {report['warning_issues']}")
    print("\n" + "=" * 60)
    
    return all_reports


if __name__ == "__main__":
    validate_all_sources()
