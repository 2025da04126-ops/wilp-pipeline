# RecoMart: End-to-End Data Management Pipeline for a Recommendation System

## Project Title
**Data Management Pipeline for Product Recommendation System - RecoMart**

## Team Member Details
- Student Name: [Your Name]
- BITS ID: [Your BITS ID]
- Course: Data Management for Machine Learning (AIMLCZG529/DSECLZG529) - S2-25

## Problem Statement
RecoMart, an e-commerce startup, needs a scalable and maintainable data management pipeline to continuously process incoming data, curate features, and train/update models for generating personalized product recommendations to improve customer engagement and cross-selling opportunities.

### Business Context
The platform collects user behavior data from multiple sources:
- Web and mobile clickstream logs (15,000+ events)
- Transactional purchase history (5,000+ transactions)
- Product metadata from catalogs (200 products, 8 categories)
- External APIs (sentiment scores, popularity metrics)

### Expected Outputs
- Clean datasets for Exploratory Data Analysis (EDA)
- Engineered features for collaborative/content-based models
- Deployable recommendation model with inference capability
- Automated, reproducible pipeline with monitoring

## Objectives
1. Build an automated, modular data pipeline supporting batch and near-real-time ingestion
2. Ensure data quality through profiling and validation (quality scoring system)
3. Engineer features suitable for collaborative and content-based filtering
4. Train and evaluate recommendation models (SVD + Content-Based)
5. Orchestrate the entire pipeline for reproducibility with Prefect/DAG

## Project Structure
```
RecoMart-Pipeline/
├── data/
│   ├── raw/                    # Raw ingested data
│   │   ├── clickstream/        # User clickstream logs (15K events)
│   │   ├── transactions/       # Purchase history (5K+ records)
│   │   ├── products/           # Product catalog + User data
│   │   └── external_api/       # External API data (sentiment/popularity)
│   ├── processed/              # Cleaned and prepared data
│   │   └── plots/              # EDA visualizations
│   ├── features/               # Engineered features + SQL schema
│   ├── feature_store/          # Versioned feature snapshots
│   └── models/                 # Trained models + metadata
├── src/
│   ├── config.py               # Central configuration
│   ├── logger.py               # Logging setup
│   ├── generate_data.py        # Synthetic data generator
│   ├── ingestion/              # Data ingestion with retry logic
│   │   └── ingest_data.py
│   ├── validation/             # Data quality checks & scoring
│   │   └── validate_data.py
│   ├── preparation/            # Data cleaning & EDA
│   │   └── prepare_data.py
│   ├── transformation/         # Feature engineering
│   │   └── transform_data.py
│   ├── feature_store/          # Feature store with versioning
│   │   └── feature_store.py
│   ├── training/               # Model training & evaluation
│   │   └── train_model.py
│   └── orchestration/          # Pipeline DAG orchestration
│       └── pipeline_dag.py
├── notebooks/                  # Jupyter notebooks for EDA
├── logs/                       # Pipeline execution logs
├── docs/                       # Documentation
├── .dvc/                       # DVC configuration
├── dvc.yaml                    # DVC pipeline stages
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Setup Instructions
```bash
# Clone the repository
git clone <repo-url>
cd RecoMart-Pipeline

# Install dependencies
pip install -r requirements.txt

# (Optional) Initialize DVC
dvc init
```

## Running the Pipeline

### Complete End-to-End Pipeline
```bash
python src/orchestration/pipeline_dag.py
```

### Individual Components
```bash
# Step 1: Generate synthetic data
python src/generate_data.py

# Step 2: Ingest data from sources
python src/ingestion/ingest_data.py

# Step 3: Validate data quality
python src/validation/validate_data.py

# Step 4: Clean and prepare data + EDA
python src/preparation/prepare_data.py

# Step 5: Feature engineering
python src/transformation/transform_data.py

# Step 6: Populate feature store
python src/feature_store/feature_store.py

# Step 7: Train and evaluate models
python src/training/train_model.py
```

### Using DVC
```bash
dvc repro  # Reproduce the entire pipeline
dvc dag    # Visualize the pipeline DAG
```

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │────>│    Ingestion    │────>│   Validation    │
│  (CSV + API)    │     │  (Retry Logic)  │     │ (Quality Score) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Preparation   │────>│ Transformation  │────>│  Feature Store  │
│  (Clean + EDA)  │     │ (Engineering)   │     │  (Versioned)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Model Training  │────>│   Evaluation    │
│  (SVD + CB)     │     │ (P@K, R@K, NDCG)│
└─────────────────┘     └─────────────────┘
```

## Data Sources

| Source | Type | Records | Description |
|--------|------|---------|-------------|
| Clickstream | CSV | 15,000 | User browsing events (views, cart, search) |
| Transactions | CSV | 5,050 | Purchase history with ratings |
| Products | CSV | 200 | Product catalog (8 categories) |
| Users | CSV | 500 | User demographics |
| External API | REST/CSV | 200 | Sentiment & popularity scores |

## Feature Engineering

### User Features (25 features)
- Transaction frequency, recency, monetary value
- Engagement score (from clickstream)
- Conversion rate, diversity of purchases
- Demographics (age, location, premium status)

### Item Features (26 features)
- Purchase popularity, unique buyers
- Average rating, review count
- View-to-purchase ratio
- External sentiment and trending scores
- Category and brand information

### Interaction Features
- User-item rating, interaction count
- Implicit feedback (views, cart adds, wishlists)
- Composite interaction score

## Models

### 1. Collaborative Filtering (SVD)
- **Algorithm**: Truncated SVD with 50 latent factors
- **Input**: User-Item interaction matrix
- **Approach**: Matrix factorization to learn latent user/item representations

### 2. Content-Based Filtering
- **Algorithm**: Cosine similarity on item features
- **Input**: Item feature vectors (21 numeric features)
- **Approach**: Recommend items similar to user's historical preferences

## Evaluation Metrics

| Metric | CF (SVD) | Content-Based |
|--------|----------|---------------|
| Precision@10 | 0.034 | 0.045 |
| Recall@10 | 0.043 | 0.068 |
| NDCG@10 | 0.033 | 0.049 |
| RMSE | 3.190 | - |

*Note: Metrics are on synthetic data. Real-world data would yield different results.*

## Data Quality

The validation module produces quality scores (0-100) for each data source:
- **Transactions**: 82/100 (detected duplicates and missing values)
- **Products**: 100/100
- **Clickstream**: 100/100
- **External Scores**: 100/100

## Data Versioning

- **DVC** is configured for tracking raw and processed data versions
- Feature store maintains versioned snapshots with metadata registry
- Model metadata (parameters, metrics, run IDs) tracked in JSON format

## Key Technologies
- **Python** (pandas, numpy, scikit-learn, scipy)
- **Visualization**: matplotlib, seaborn
- **Orchestration**: Prefect / Custom DAG
- **Versioning**: DVC, Git
- **Feature Store**: Custom implementation with versioned snapshots
- **Model Tracking**: MLflow-style JSON metadata

## Conclusion
This project demonstrates a production-ready data management pipeline covering all stages from ingestion to model deployment, with proper versioning, validation, and orchestration. The modular architecture allows independent development and testing of each component while maintaining end-to-end reproducibility through the DAG orchestrator.

### Future Scope
- Integration with real-time streaming (Apache Kafka)
- Cloud deployment (AWS S3, SageMaker)
- A/B testing framework for model comparison
- Deep learning models (Neural Collaborative Filtering)
- Real-time inference API endpoint
