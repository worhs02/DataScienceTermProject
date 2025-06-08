from operator import is_
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA




# Import Data
df = pd.read_csv("screen_time_app_usage_dataset.csv")

# Convert Date Format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Fill with missing values of 0 for YouTube
youtube_cols = ['youtube_views', 'youtube_likes', 'youtube_comments']
df[youtube_cols] = df[youtube_cols].fillna(0)

# add is_entertainment feature
df['is_entertainment'] = (df['category'] == 'Entertainment').astype(int)

# remove duplication
df = df.drop_duplicates(subset=['user_id', 'date', 'app_name'])

# Total daily usage hours by user
total_usage = df.groupby(['user_id', 'date'])['screen_time_min'].sum().reset_index(name='total_usage')

# Productivity app usage time
productivity_usage = df[df['category'] == 'Productivity'].groupby(['user_id', 'date'])['screen_time_min'].sum().reset_index(name='productivity_usage')

# Entertainment App Usage Time
entertainment_usage = df[df['category'].isin(['Entertainment', 'Social'])].groupby(['user_id', 'date'])['screen_time_min'].sum().reset_index(name='entertainment_usage')

# YouTube Usage time
youtube_df = df[df['app_name'].str.lower() == 'youtube']
youtube_usage = youtube_df.groupby(['user_id', 'date'])['screen_time_min'].sum().reset_index(name='youtube_usage')

# Number of interactions
interactions = df.groupby(['user_id', 'date'])['interactions'].sum().reset_index(name='total_interactions')

is_entertainment = df.groupby(['user_id', 'date'])['is_entertainment'].sum().reset_index(name='is_entertainment')

# merge
merged = total_usage \
    .merge(productivity_usage, on=['user_id', 'date'], how='left') \
    .merge(entertainment_usage, on=['user_id', 'date'], how='left') \
    .merge(youtube_usage, on=['user_id', 'date'], how='left') \
    .merge(interactions, on=['user_id', 'date'], how='left') \
    .merge(is_entertainment, on=['user_id', 'date'], how='left')

# Missing value processing
merged.fillna(0, inplace=True)

# Generating Derivative Features
merged['productivity_ratio'] = merged['productivity_usage'] / (merged['total_usage'] + 1e-5)
merged['entertainment_ratio'] = merged['entertainment_usage'] / (merged['total_usage'] + 1e-5)

# Scaling numerical variables
scaler = StandardScaler()
numerical_cols = ['total_usage', 'productivity_usage', 'entertainment_usage', 'youtube_usage', 'total_interactions']
merged_scaled = merged.copy()
merged_scaled[numerical_cols] = scaler.fit_transform(merged[numerical_cols])

# Final Feature Data
feature_df = merged_scaled

# Create User Type: productivity_ratio > 0.5 → Productivity user (1), or entertainment user (0)
feature_df['user_type'] = (feature_df['productivity_ratio'] > 0.5).astype(int)

# Define Input Features (X) and Target (y)
X = feature_df[['productivity_ratio', 'entertainment_ratio', 'total_usage', 'total_interactions']]
y = feature_df['user_type']

# data segmentation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition and Learning
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Performance Evaluation Output
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# cross-validation
cv_score = cross_val_score(model, X, y, cv=5).mean()
print(f"\nCross-validation Accuracy: {cv_score:.3f}")

# Sort by user_id and date
feature_df = feature_df.sort_values(by=['user_id', 'date'])

# Create YouTube usage time the next day
# Shift 'youtube_usage' back 1 day per user_id to create 'next_day_youtube_usage' feature
feature_df['next_day_youtube_usage'] = feature_df.groupby('user_id')['youtube_usage'].shift(-1)

# Create tagged features for prediction (e.g., YouTube usage time the day before)
feature_df['prev_day_youtube_usage'] = feature_df.groupby('user_id')['youtube_usage'].shift(1)

# Missing value treatment (NaN value due to shift)
# Rows with next_day_youtube_usage NaN are last-day data and are excluded from prediction
feature_df.dropna(subset=['next_day_youtube_usage'], inplace=True)
#A row with prev_day_youtube_usage NaN is the first day data for that user, so fill it with zero or exclude it
feature_df['prev_day_youtube_usage'] = feature_df['prev_day_youtube_usage'].fillna(0)

# Define features (X) and targets (y) for prediction
# Use existing features and newly created 'prev_day_youtube_usage'
X = feature_df[['productivity_ratio', 'entertainment_ratio', 'total_usage',
                'total_interactions', 'prev_day_youtube_usage']]
y = feature_df['next_day_youtube_usage']

# data segmentation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


# Define each model
models = {
    "Linear Regression": LinearRegression(),
     "Random Forest (n=200, depth=10)": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Random Forest (n=100, depth=None)": RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
    "Random Forest (n=100, depth=5)": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "Random Forest (n=50, depth=3)": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
}

results = []


# Learn and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "model": name,
        "mse": mse,
        "r2": r2,
        "y_pred": y_pred
    })
    print(f"{name}\nMSE: {mse:.4f}, R²: {r2:.4f}\n")

# Select the best performing model (selected by R² here)
best_result = max(results, key=lambda x: x["r2"])
best_model_name = best_result["model"]
best_y_pred = best_result["y_pred"]

print(f"Best Model: {best_model_name} (R²: {best_result['r2']:.4f})")

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(y_test, best_y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True YouTube Usage Time")
plt.ylabel("Predicted YouTube Usage Time")
plt.title(f"Best Model ({best_model_name}): True vs. Predicted YouTube Usage")
plt.grid(True)
plt.tight_layout()
plt.show()


# Select features for clustering
cluster_features = feature_df[['productivity_ratio', 'entertainment_ratio', 'youtube_usage', 'total_usage']]

# Remove missing values
cluster_features = cluster_features.dropna()

# k-means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
feature_df['cluster'] = kmeans.fit_predict(cluster_features)

# PCA 2D reduction
pca = PCA(n_components=2)
cluster_pca = pca.fit_transform(cluster_features)


plt.figure(figsize=(10, 6))
plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], c=feature_df['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('User Clusters based on App Usage')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# YouTube Overindulgence Detection
# Calculate mean and standard deviation by user
user_stats = feature_df.groupby('user_id')['youtube_usage'].agg(['mean', 'std']).reset_index()
user_stats.rename(columns={'mean': 'youtube_mean', 'std': 'youtube_std'}, inplace=True)

# Merge user-specific mean and standard deviation to existing feature_df
feature_df = feature_df.merge(user_stats, on='user_id', how='left')

# Calculation of over-immersion threshold: mean + 2 * standard deviation
feature_df['youtube_threshold'] = feature_df['youtube_mean'] + 2 * feature_df['youtube_std']

# Determination of over-indulgence
feature_df['is_overused'] = (feature_df['youtube_usage'] > feature_df['youtube_threshold']).astype(int)

# Filter only days when you're over-involved
overused_days = feature_df[feature_df['is_overused'] == 1][['user_id', 'date', 'youtube_usage', 'youtube_threshold']]

# Output Results
print("\nOver-immersed Day List (by YouTube use):")
print(overused_days.sort_values(['user_id', 'date']).head(10))

# Visualization of Over-immersive Heatmaps
plt.figure(figsize=(12, 6))
sns.heatmap(
    feature_df.pivot_table(index='user_id', columns='date', values='is_overused', fill_value=0),
    cmap='Reds', cbar_kws={'label': 'Overuse Detected'}
)
plt.title('YouTube Overuse Detection Heatmap')
plt.xlabel('Date')
plt.ylabel('User ID')
plt.tight_layout()
plt.show()

# Calculation of over-immersion ratio by cluster
cluster_overuse_ratio = feature_df.groupby('cluster')['is_overused'].mean().reset_index()
cluster_overuse_ratio['overuse_ratio_percent'] = cluster_overuse_ratio['is_overused'] * 100

# Filter cluster 0 and 1 only
cluster_overuse_ratio = cluster_overuse_ratio[cluster_overuse_ratio['cluster'].isin([0, 1])]

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(data=cluster_overuse_ratio, x='cluster', y='overuse_ratio_percent')
plt.title('Cluster-wise YouTube Overuse Rate (Only Clusters 0 & 1)')
plt.xlabel('Cluster')
plt.ylabel('Overuse Ratio (%)')
plt.ylim(0, 50)
plt.grid(True, axis='y', linestyle='-', alpha=0.6)
plt.tight_layout()
plt.show()
