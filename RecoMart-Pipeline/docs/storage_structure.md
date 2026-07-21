# Storage Structure Documentation - RecoMart Pipeline

## Overview
The RecoMart pipeline uses a structured local filesystem layout organized by 
data source, processing stage, and type. This document describes the storage 
architecture.

## Directory Layout

### Raw Data Storage (`data/raw/`)
Raw data is partitioned by source type:

```
data/raw/
├── clickstream/
│   └── clickstream.csv          # 15,000 user browsing events
├── transactions/
│   └── transactions.csv         # 5,050 purchase records
├── products/
│   ├── products.csv             # 200 product catalog entries
│   └── users.csv                # 500 user demographics
├── external_api/
│   └── external_scores.csv      # External sentiment/popularity data
└── ingestion_report.json        # Ingestion audit log
```

### Processed Data (`data/processed/`)
Cleaned and prepared datasets ready for feature engineering:

```
data/processed/
├── transactions_clean.csv       # Deduplicated, imputed transactions
├── products_clean.csv           # Normalized product data
├── users_clean.csv              # Encoded user demographics
├── clickstream_clean.csv        # Cleaned browsing events
├── external_scores_clean.csv    # Validated external data
├── data_quality_report.json     # Validation results
└── plots/                       # EDA visualizations
    ├── eda_overview.png
    ├── interaction_heatmap.png
    └── temporal_patterns.png
```

### Engineered Features (`data/features/`)
Feature vectors ready for model consumption:

```
data/features/
├── user_features.csv            # 500 users × 25 features
├── item_features.csv            # 200 items × 26 features
├── interaction_features.csv     # 18,123 user-item pairs
├── feature_metadata.json        # Feature column documentation
└── feature_schema.sql           # SQL DDL for feature tables
```

### Feature Store (`data/feature_store/`)
Versioned feature snapshots:

```
data/feature_store/
├── feature_registry.json        # Feature metadata registry
└── versions/
    └── v{timestamp}/            # Versioned snapshot
        ├── user_features.csv
        ├── item_features.csv
        └── interaction_features.csv
```

### Models (`data/models/`)
Trained model artifacts and tracking:

```
data/models/
├── cf_svd_model.pkl            # Collaborative Filtering (SVD)
├── cb_model.pkl                # Content-Based Filtering
└── model_metadata.json         # MLflow-style run tracking
```

## Data Schemas

### Clickstream Events
| Column | Type | Description |
|--------|------|-------------|
| event_id | str | Unique event identifier |
| user_id | str | User identifier |
| product_id | str | Product identifier |
| event_type | str | page_view, product_view, add_to_cart, etc. |
| session_id | str | Browsing session ID |
| timestamp | datetime | Event timestamp |
| device | str | mobile, desktop, tablet |
| page_duration_sec | int | Time spent on page |

### Transactions
| Column | Type | Description |
|--------|------|-------------|
| transaction_id | str | Unique transaction ID |
| user_id | str | Purchasing user |
| product_id | str | Purchased product |
| quantity | int | Units purchased |
| rating | float | User rating (1-5) |
| timestamp | datetime | Purchase time |
| payment_method | str | Payment type |

### Products
| Column | Type | Description |
|--------|------|-------------|
| product_id | str | Unique product ID |
| product_name | str | Product name |
| category | str | Product category |
| price | float | Price in INR |
| brand | str | Brand name |
| avg_rating | float | Average rating |
| num_reviews | int | Review count |
| in_stock | bool | Availability |

### Users
| Column | Type | Description |
|--------|------|-------------|
| user_id | str | Unique user ID |
| age | int | User age |
| gender | str | M, F, Other |
| location | str | City |
| signup_date | date | Registration date |
| is_premium | bool | Premium membership |

## Naming Conventions
- Raw files: `{source_name}.csv`
- Processed files: `{source_name}_clean.csv`
- Feature files: `{entity}_features.csv`
- Versioned files: `v{YYYYMMDD_HHMMSS}/`
- Logs: `pipeline_{YYYYMMDD}.log`
