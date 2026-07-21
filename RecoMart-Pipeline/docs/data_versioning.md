# Data Versioning Workflow - RecoMart Pipeline

## Overview
This document describes the data versioning strategy used in the RecoMart pipeline 
to track changes across raw data, processed data, features, and models.

## Tools Used
- **DVC (Data Version Control)**: For versioning large data files alongside Git
- **Custom Feature Store**: For versioned feature snapshots with metadata registry

## DVC Setup

### Configuration
```
.dvc/config:
  remote = local_storage (can be changed to S3/GCS)
  autostage = true
```

### Pipeline Stages (dvc.yaml)
1. `generate_data` - Creates synthetic raw data
2. `ingest` - Ingests from sources with metadata
3. `validate` - Runs quality checks, produces quality report
4. `prepare` - Cleans and preprocesses data
5. `transform` - Engineers features
6. `feature_store` - Populates versioned feature store
7. `train` - Trains and evaluates models

### Usage
```bash
# Reproduce full pipeline
dvc repro

# Reproduce specific stage
dvc repro train

# View pipeline DAG
dvc dag

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

## Feature Store Versioning

### Structure
```
data/feature_store/
├── feature_registry.json    # Metadata for all features
└── versions/
    ├── v20260720_205300/    # Version snapshots
    │   ├── user_features.csv
    │   ├── item_features.csv
    │   └── interaction_features.csv
    └── ...
```

### Feature Registry Schema
Each feature is registered with:
- **name**: Feature identifier
- **description**: Human-readable description
- **source**: Data source (transactions, clickstream, external_api)
- **transformation**: Applied transformation logic
- **dtype**: Data type
- **entity**: Entity type (user, item, interaction)
- **created_at**: Registration timestamp

### Version Retrieval
```python
from src.feature_store.feature_store import FeatureStore

store = FeatureStore()
store.ingest_features()

# Get latest version
features = store.get_features_for_training()

# Get specific version
features = store.get_features_for_training("v20260720_205300")

# Get single entity features (for inference)
user_feats = store.get_user_features("U0001")
item_feats = store.get_item_features("P0001")
```

## Model Versioning

### Model Metadata Tracking
Each model run saves:
- `run_id`: Unique identifier based on timestamp
- Algorithm parameters (n_factors, similarity method)
- Evaluation metrics (Precision@K, Recall@K, NDCG, RMSE)
- Training parameters (test_size, random_state)

### Storage
```
data/models/
├── cf_svd_model.pkl         # Collaborative Filtering model
├── cb_model.pkl             # Content-Based model
└── model_metadata.json      # Run tracking metadata
```

## Data Lineage

### Tracking Metadata
Each data file maintains lineage through:
1. **Ingestion timestamp**: When data was collected
2. **Source file/API**: Origin of the data
3. **Transformations applied**: Cleaning, encoding, normalization steps
4. **Version hash**: MD5 hash for integrity verification

### Example Lineage
```
Raw Transactions (CSV)
  → Ingested (added metadata columns)
    → Validated (quality score: 82/100)
      → Cleaned (removed 47 duplicates, imputed missing)
        → Features (user_features, item_features, interactions)
          → Model Training (SVD with 50 factors)
```
