
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
