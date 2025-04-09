from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, expr, lit
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.clustering import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure Spark for a laptop with 8GB RAM
def configure_spark_for_local_machine():
    spark = SparkSession.builder \
        .appName("ArgoFloatAnalysis") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "1g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .getOrCreate()
    
    # Reduce log output
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# Initialize Spark session
spark = configure_spark_for_local_machine()
print("Spark session initialized successfully")

# Load CSV data with explicit schema inference
print("Loading Argo float data...")
df = spark.read.option("header", "true").csv("data.csv")

# Print original schema to see data types
print("Original Schema:")
df.printSchema()

# Check for missing values before conversion
print("\nChecking for missing or empty values in key columns:")
df.select([
    count(when(col(c).isNull() | (col(c) == ""), 1)).alias(c) 
    for c in ["latitude", "longitude", "pres_adjusted", "temp_adjusted"]
]).show()

# Convert string columns to double type
print("\nConverting string columns to appropriate numeric types...")
df = df.withColumn("latitude", col("latitude").cast(DoubleType())) \
       .withColumn("longitude", col("longitude").cast(DoubleType())) \
       .withColumn("pres_adjusted", col("pres_adjusted").cast(DoubleType())) \
       .withColumn("temp_adjusted", col("temp_adjusted").cast(DoubleType()))

# Check schema after conversion
print("Schema after conversion:")
df.printSchema()

# Check for NaN values after conversion
print("\nChecking for NaN values after type conversion:")
df.select([
    count(when(isnan(col(c)), 1)).alias(c) 
    for c in ["latitude", "longitude", "pres_adjusted", "temp_adjusted"]
]).show()

# Filter to keep only high-quality data using the quality flags
print("\nFiltering data based on quality flags...")
if "pres_adjusted_qc" in df.columns and "temp_adjusted_qc" in df.columns:
    df_filtered = df.filter((col("pres_adjusted_qc") == '1') & (col("temp_adjusted_qc") == '1'))
    print(f"Data after quality filtering: {df_filtered.count()} rows")
else:
    df_filtered = df
    print("Quality flag columns not found in expected format, using all data")

# Select necessary columns and handle missing values
print("\nSelecting key columns and handling missing values...")
df_selected = df_filtered.select("time", "latitude", "longitude", "pres_adjusted", "temp_adjusted")
df_clean = df_selected.na.drop()
print(f"Data after removing null values: {df_clean.count()} rows")

# Show summary statistics
print("\nSummary statistics for clean data:")
df_clean.describe().show()

# Create a temporary view for SQL analysis
df_clean.createOrReplaceTempView("argo_data")

# SQL-based insights
print("\nAverage temperature by depth range (SQL):")
spark.sql("""
    SELECT 
        CASE 
            WHEN pres_adjusted < 100 THEN 'Surface (<100)'
            WHEN pres_adjusted >= 100 AND pres_adjusted < 500 THEN 'Mid (100-500)'
            WHEN pres_adjusted >= 500 AND pres_adjusted < 1000 THEN 'Deep (500-1000)'
            ELSE 'Very Deep (>1000)'
        END as depth_range,
        AVG(temp_adjusted) as avg_temp,
        COUNT(*) as measurement_count
    FROM argo_data
    GROUP BY 1
    ORDER BY 
        CASE 
            WHEN depth_range = 'Surface (<100)' THEN 1
            WHEN depth_range = 'Mid (100-500)' THEN 2
            WHEN depth_range = 'Deep (500-1000)' THEN 3
            ELSE 4
        END
""").show()

# Visualizations with sample data
try:
    # Use a sample for visualization to avoid memory issues
    print("\nGenerating sample data for visualizations...")
    sample_size = min(10000, df_clean.count())
    df_sample = df_clean.limit(sample_size).toPandas()
    
    print(f"Successfully collected {len(df_sample)} samples for visualization")
    
    # Create directory for saving visualizations
    import os
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    
    # Visualization 1: Temperature distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df_sample['temp_adjusted'], kde=True)
    plt.title('Distribution of Ocean Temperatures')
    plt.xlabel('Temperature (°C)')
    plt.savefig('visualizations/temp_distribution.png')
    plt.close()
    
    # Visualization 2: Relationship between pressure and temperature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pres_adjusted', y='temp_adjusted', data=df_sample, alpha=0.5)
    plt.title('Relationship between Pressure and Temperature')
    plt.xlabel('Pressure (dbar)')
    plt.ylabel('Temperature (°C)')
    plt.savefig('visualizations/pressure_temp_relationship.png')
    plt.close()
    
    # Visualization 3: Geographical distribution of measurements
    plt.figure(figsize=(12, 8))
    plt.scatter(df_sample['longitude'], df_sample['latitude'], 
              c=df_sample['temp_adjusted'], cmap='viridis', 
              alpha=0.7, s=15)
    plt.colorbar(label='Temperature (°C)')
    plt.title('Geographical Distribution of Ocean Temperatures')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('visualizations/geo_distribution.png')
    plt.close()
    
    print("Visualization plots saved to 'visualizations' directory")
except Exception as e:
    print(f"Error generating visualizations: {e}")

# Feature engineering
print("\nPerforming feature engineering...")

# Add derived features
df_ml = df_clean.withColumn("depth_approx", col("pres_adjusted") / 10)  # Rough approximation

# Calculate mean temperature for classification threshold
mean_temp = df_ml.select(expr("avg(temp_adjusted)")).collect()[0][0]
print(f"Mean temperature threshold for classification: {mean_temp:.4f}°C")

# Create binary label for classification: 1 for warm (above mean), 0 for cold (below mean)
df_labeled = df_ml.withColumn("label", when(col("temp_adjusted") > mean_temp, 1.0).otherwise(0.0))

# Split data for training and testing (70% training, 30% testing)
splits = df_labeled.randomSplit([0.7, 0.3], seed=42)
train_data = splits[0]
test_data = splits[1]

print(f"Training set size: {train_data.count()} rows")
print(f"Test set size: {test_data.count()} rows")

# Machine Learning - Classification
print("\n===== CLASSIFICATION MODELS =====")
print("Predicting whether water is warm or cold based on location and pressure")

# Prepare features for classification
classification_cols = ["latitude", "longitude", "pres_adjusted", "depth_approx"]
assembler = VectorAssembler(inputCols=classification_cols, outputCol="features_unscaled")

# Apply VectorAssembler
train_assembled = assembler.transform(train_data)
test_assembled = assembler.transform(test_data)

# Standardize features
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", 
                      withStd=True, withMean=True)
scaler_model = scaler.fit(train_assembled)
train_scaled = scaler_model.transform(train_assembled)
test_scaled = scaler_model.transform(test_assembled)

# Train logistic regression model
print("\nTraining Logistic Regression classifier...")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_scaled)

# Make predictions
lr_predictions = lr_model.transform(test_scaled)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(lr_predictions)
print(f"Logistic Regression AUC: {auc:.4f}")

# Train Random Forest classifier
print("\nTraining Random Forest classifier...")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
rf_model = rf.fit(train_scaled)

# Make predictions
rf_predictions = rf_model.transform(test_scaled)

# Evaluate Random Forest model
rf_auc = evaluator.evaluate(rf_predictions)
print(f"Random Forest AUC: {rf_auc:.4f}")

# Feature importance for Random Forest
if hasattr(rf_model, "featureImportances"):
    feature_importance = rf_model.featureImportances.toArray()
    feature_names = classification_cols
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })
    print("\nFeature importance (Random Forest):")
    for i, row in importance_df.sort_values("Importance", ascending=False).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.6f}")

# Machine Learning - Regression
print("\n===== REGRESSION MODELS =====")
print("Predicting ocean temperature based on location and pressure")

# Prepare features for regression
regression_cols = ["latitude", "longitude", "pres_adjusted"]
reg_assembler = VectorAssembler(inputCols=regression_cols, outputCol="features_unscaled")

# Apply VectorAssembler
reg_train_assembled = reg_assembler.transform(train_data)
reg_test_assembled = reg_assembler.transform(test_data)

# Standardize features
reg_scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", 
                         withStd=True, withMean=True)
reg_scaler_model = reg_scaler.fit(reg_train_assembled)
reg_train_scaled = reg_scaler_model.transform(reg_train_assembled)
reg_test_scaled = reg_scaler_model.transform(reg_test_assembled)

# Train linear regression model
print("\nTraining Linear Regression model...")
lr_reg = LinearRegression(featuresCol="features", labelCol="temp_adjusted", maxIter=10)
lr_reg_model = lr_reg.fit(reg_train_scaled)

# Make predictions
lr_reg_predictions = lr_reg_model.transform(reg_test_scaled)

# Evaluate the model
reg_evaluator = RegressionEvaluator(labelCol="temp_adjusted", predictionCol="prediction")
rmse = reg_evaluator.evaluate(lr_reg_predictions)
r2 = reg_evaluator.setMetricName("r2").evaluate(lr_reg_predictions)
print(f"Linear Regression RMSE: {rmse:.4f}")
print(f"Linear Regression R²: {r2:.4f}")

# Print model coefficients
print("\nLinear Regression coefficients:")
coef_list = list(zip(regression_cols, lr_reg_model.coefficients.toArray()))
for name, coef in coef_list:
    print(f"{name}: {coef:.6f}")
print(f"Intercept: {lr_reg_model.intercept:.6f}")

# Train Random Forest regressor
print("\nTraining Random Forest regression model...")
rf_reg = RandomForestRegressor(featuresCol="features", labelCol="temp_adjusted", numTrees=10)
rf_reg_model = rf_reg.fit(reg_train_scaled)

# Make predictions
rf_reg_predictions = rf_reg_model.transform(reg_test_scaled)

# Evaluate Random Forest regression model
rf_rmse = reg_evaluator.setMetricName("rmse").evaluate(rf_reg_predictions)
rf_r2 = reg_evaluator.setMetricName("r2").evaluate(rf_reg_predictions)
print(f"Random Forest Regression RMSE: {rf_rmse:.4f}")
print(f"Random Forest Regression R²: {rf_r2:.4f}")

# Feature importance for Random Forest regressor
if hasattr(rf_reg_model, "featureImportances"):
    reg_importance = rf_reg_model.featureImportances.toArray()
    reg_feature_names = regression_cols
    reg_importance_df = pd.DataFrame({
        "Feature": reg_feature_names,
        "Importance": reg_importance
    })
    print("\nFeature importance (Random Forest Regression):")
    for i, row in reg_importance_df.sort_values("Importance", ascending=False).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.6f}")

# Advanced Analysis: Clustering to identify water masses
print("\n===== ADVANCED ANALYSIS: WATER MASS CLUSTERING =====")
print("Using K-means clustering to identify distinct water masses")

# Prepare features for clustering
cluster_cols = ["latitude", "longitude", "pres_adjusted", "temp_adjusted"]
cluster_assembler = VectorAssembler(inputCols=cluster_cols, outputCol="features_unscaled")

# Sample data for clustering (to ensure it runs on limited RAM)
cluster_sample = df_clean.sample(False, min(50000, df_clean.count()) / df_clean.count(), seed=42)
print(f"Using {cluster_sample.count()} samples for clustering analysis")

# Apply VectorAssembler
cluster_assembled = cluster_assembler.transform(cluster_sample)

# Standardize features
cluster_scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", 
                             withStd=True, withMean=True)
cluster_scaler_model = cluster_scaler.fit(cluster_assembled)
cluster_scaled = cluster_scaler_model.transform(cluster_assembled)

# Find optimal number of clusters (or use domain knowledge)
k = 4  # Starting with 4 clusters, can be optimized

# Apply K-means clustering
kmeans = KMeans(featuresCol="features", k=k, seed=42)
kmeans_model = kmeans.fit(cluster_scaled)

# Get cluster centers
centers = kmeans_model.clusterCenters()
print(f"\nCluster Centers (in standardized feature space):")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

# Apply model to get cluster assignments
clustered_data = kmeans_model.transform(cluster_scaled)

# Convert to Pandas for analysis by cluster
try:
    pd_clusters = clustered_data.select("latitude", "longitude", "pres_adjusted", "temp_adjusted", "prediction").toPandas()
    
    # Summarize clusters
    cluster_summary = pd_clusters.groupby("prediction").agg({
        "latitude": "mean",
        "longitude": "mean",
        "pres_adjusted": "mean",
        "temp_adjusted": "mean",
        "prediction": "count"
    }).rename(columns={"prediction": "count"}).reset_index()
    
    print("\nCluster Summary:")
    print(cluster_summary)
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="longitude", y="latitude", hue="prediction", data=pd_clusters, palette="viridis", alpha=0.7)
    plt.title("Geographical Distribution of Water Mass Clusters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("visualizations/water_mass_clusters.png")
    
    # Temperature vs Pressure by cluster
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="pres_adjusted", y="temp_adjusted", hue="prediction", data=pd_clusters, palette="viridis", alpha=0.7)
    plt.title("Temperature vs Pressure by Water Mass Cluster")
    plt.xlabel("Pressure (dbar)")
    plt.ylabel("Temperature (°C)")
    plt.savefig("visualizations/temp_pressure_clusters.png")
    plt.close()
    
    print("Cluster visualization saved to 'visualizations' directory")
except Exception as e:
    print(f"Error generating cluster visualizations: {e}")

# Time Series Analysis (if data has sufficient time coverage)
print("\n===== SIMPLE TIME SERIES ANALYSIS =====")
try:
    # Extract year and month from the time column
    time_df = df_clean.withColumn("year_month", expr("substring(time, 1, 7)"))
    
    # Create a temporary view
    time_df.createOrReplaceTempView("time_series")
    
    # Calculate average temperature by year-month
    time_series_data = spark.sql("""
        SELECT 
            year_month,
            COUNT(*) as measurement_count,
            AVG(temp_adjusted) as avg_temp
        FROM time_series
        GROUP BY year_month
        ORDER BY year_month
    """).cache()
    
    print("Monthly temperature averages (sample):")
    time_series_data.show(5)
    
    # Convert to pandas for visualization
    if time_series_data.count() > 0:
        time_series_pd = time_series_data.toPandas()
        
        # Plot time series
        plt.figure(figsize=(15, 6))
        plt.plot(time_series_pd['year_month'], time_series_pd['avg_temp'], marker='o')
        plt.title('Average Ocean Temperature Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Average Temperature (°C)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('visualizations/temp_time_series.png')
        plt.close()
        
        print("Time series visualization saved to 'visualizations' directory")
    else:
        print("Insufficient time series data for visualization")
        
except Exception as e:
    print(f"Error in time series analysis: {e}")

# Summarize findings
print("\n===== SUMMARY OF FINDINGS =====")
print("1. Data Quality Assessment:")
print(f"   - Total valid data points: {df_clean.count()}")
print(f"   - Temperature range: {df_clean.agg({'temp_adjusted': 'min'}).collect()[0][0]:.2f}°C to {df_clean.agg({'temp_adjusted': 'max'}).collect()[0][0]:.2f}°C")
print(f"   - Mean temperature: {mean_temp:.2f}°C")

print("\n2. Classification Models:")
print(f"   - Logistic Regression AUC: {auc:.4f}")
print(f"   - Random Forest AUC: {rf_auc:.4f}")
print(f"   - Best classifier: {'Random Forest' if rf_auc > auc else 'Logistic Regression'}")

print("\n3. Regression Models:")
print(f"   - Linear Regression RMSE: {rmse:.4f}, R²: {r2:.4f}")
print(f"   - Random Forest RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")
print(f"   - Best regression model: {'Random Forest' if rf_r2 > r2 else 'Linear Regression'}")

print("\n4. Clustering Analysis:")
print(f"   - {k} distinct water masses identified")
print("   - See cluster visualizations for spatial distribution")

print("\nAnalysis complete. All outputs saved to 'visualizations' directory.")

# Stop Spark session
spark.stop()
print("Spark session stopped.")