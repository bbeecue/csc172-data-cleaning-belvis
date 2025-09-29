# Data Cleaning with AI Support

## Student Information
- Name: FEBE GWYN R. BELVIS
- Course Year: BSCS 4
- Date: 2004-02-02

## Dataset
- Source: https://archive.ics.uci.edu/dataset/2/adult
- Name: "Census Income" dataset

## Issues found
- Missing values: 
age                  0
workclass         1836
fnlwgt               0
education            0
education-num        0
marital-status       0
occupation        1843
relationship         0
race                 0
sex                  0
capital-gain         0
capital-loss         0
hours-per-week       0
native-country     583
income               0

- Duplicates: 
24


- Inconsistencies:  ...

## Cleaning steps
1. Missing values: ...
2. Duplicates: ...
3. Inconsistencies: ...
4. Outliers: ...

## AI prompts used
**Prompt 1:**
"You are doing data mining and analysis using Pandas. Give me a Python code that does an exploratory check on a CSV Census dataset named "raw_dataset". Do not clean or preprocess the the dataset yet. Display the following: - Dataframe info (df.info) - Description of the dataframe (df.describe) - Missing values - Duplicates - Sample rows - Graphs and Summary statistics"

- Generated code: 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("raw_dataset.csv")

# Display basic info
print("=== Dataframe Info ===")
print(df.info())
print("\n")

# Statistical description
print("=== Dataframe Description ===")
print(df.describe(include='all'))  # include='all' covers categorical columns too
print("\n")

# Check missing values
print("=== Missing Values ===")
print(df.isnull().sum())
print("\n")

# Check duplicates
print("=== Duplicate Rows ===")
print(df.duplicated().sum())
print("\n")

# Show sample rows
print("=== Sample Rows ===")
print(df.sample(5))   # random 5 rows
print("\n")

# ---------------------------
# Graphs and Summary Statistics
# ---------------------------

# Histogram for numeric columns
df.hist(figsize=(12, 8), bins=30)
plt.suptitle("Histograms of Numeric Columns")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Count plots for categorical columns (up to 5 to avoid clutter)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:5]

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.show()

# Boxplots for detecting outliers in numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:5]
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


- Prompt 2: "The original dataset doesn't have column names or variable names. Include naming of columns in the initial exploration of the dataset after importing the csv file. 

The columns have the following names in order: 
- age 
- workclass 
- fnlwgt 
- education 
- education-num 
- marital-status 
- occupation 
- relationship 
- race 
- sex 
- capital-gain 
- capital-loss 
- hours-per-week 
- native-country 
- income

Additionally, the null items are marked as '?'."

Generated Code:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Column names based on dataset documentation
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load dataset (no headers originally, so header=None)
df = pd.read_csv(
    "raw_dataset.csv",
    header=None,          # because the file has no header row
    names=columns,        # assign the column names
    na_values="?"         # treat '?' as missing values
)

# ---------------------------
# Basic Info
# ---------------------------
print("=== Dataframe Info ===")
print(df.info())
print("\n")

print("=== Dataframe Description ===")
print(df.describe(include='all'))
print("\n")

# Missing values
print("=== Missing Values ===")
print(df.isnull().sum())
print("\n")

# Duplicates
print("=== Duplicate Rows ===")
print(df.duplicated().sum())
print("\n")

# Sample rows
print("=== Sample Rows ===")
print(df.sample(5))
print("\n")

# ---------------------------
# Graphs and Summary Statistics
# ---------------------------

# Histograms for numeric columns
df.hist(figsize=(14, 10), bins=30)
plt.suptitle("Histograms of Numeric Columns")
plt.show()

# Correlation heatmap (numeric only)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Count plots for some categorical variables
categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:5]
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.show()

# Boxplots for numeric variables
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:5]
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


## Results
- Rows before: 32561
- Rows after: M