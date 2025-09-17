# import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore Dataset

# Load Iris dataset from sklearn and convert to DataFrame
try:
    iris = load_iris(as_frame=True)
    df = iris.frame  # Includes both features and target
    df['species'] = df['target'].map(dict(enumerate(iris.target_names)))  # Map target integers to species names
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ File not found. Please check the dataset path.")

# Display first few rows
print("🔎 First 5 rows of dataset:")
print(df.head())

# Check data types and missing values
print("\n📊 Dataset Info:")
print(df.info())

print("\n❓ Missing values in dataset:")
print(df.isnull().sum())

# Clean dataset (Iris has no missing values, but we'll handle generally)
df = df.dropna()  # If missing values existed, we could also use fillna()

# Task 2: Basic Data Analysis
 
# Compute basic statistics
print("\n📈 Basic Statistics:")
print(df.describe())

# Group by species and compute mean of numerical features
print("\n📌 Mean values grouped by species:")
grouped_means = df.groupby('species').mean(numeric_only=True)
print(grouped_means)

# Identify patterns
print("\n🔍 Observations:")
print("- Setosa generally has smaller petal dimensions compared to the others.")
print("- Virginica tends to have the largest petal length and width.")
print("- Sepal sizes overlap more across species compared to petal sizes.")

# Task 3: Data Visualization

# 1. Line chart (example: petal length trend over index for first 30 rows)
plt.figure(figsize=(8,5))
plt.plot(df.index[:30], df['petal length (cm)'][:30], marker='o', label='Petal Length')
plt.title("Line Chart: Petal Length Trend (First 30 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x='species', y='petal length (cm)', data=df, estimator='mean', ci=None)
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal length)
plt.figure(figsize=(8,5))
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length, colored by species)
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='deep')
plt.title("Scatter Plot: Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
