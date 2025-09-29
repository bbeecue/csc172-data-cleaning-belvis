# Data Cleaning with AI Support

## Student Information
- Name: FEBE GWYN R. BELVIS
- Course Year: BSCS 4
- Date: 2025-09-29

## Dataset
- Source: https://archive.ics.uci.edu/dataset/2/adult
- Name: "Census Income" dataset

## Issues found
- Missing values: 
workclass         1836
occupation        1843
native-country     583

- Duplicates: 
24


- Inconsistencies:  Whitespaces before values, not standardized cases

## Cleaning steps
1. Missing values: Filled the missing numerical attributes with median value of that attribute. While for the non-numerical, we choose to fill it with mode (However this will introduce bias and distort relationships)
2. Duplicates: Dropped duplicates from the dataframe directly
3. Inconsistencies: Removed extra spaces and lowercase for all categories. Standardized also the target variable to binary format (<=50k, >50k)
4. Outliers: Classical IQR (Interquartile Range) method for detection and winsorization (capping) as outlier treatment.

## AI prompts used
**Prompt 1:**
"You are doing data mining and analysis using Pandas. Give me a Python code that does an exploratory check on a CSV Census dataset named "raw_dataset". Do not clean or preprocess the the dataset yet. Display the following: - Dataframe info (df.info) - Description of the dataframe (df.describe) - Missing values - Duplicates - Sample rows - Graphs and Summary statistics"

- Generated code: 

```
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

```

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

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Column names based on dataset documentation
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
```

- Prompt 3: Since the dataset contains numerical and non-numerical values, what should I do for the statistical summary?

Generated Answer:
```
Good question 👌

By default, pandas.DataFrame.describe() behaves differently depending on column type:

For numerical columns → it shows count, mean, std, min, quartiles, max.

For categorical (non-numerical) columns → it shows count, unique, top (most frequent), and freq (count of top).

Since your dataset has both types, you have a few options:

🔹 Option 1: Run separate summaries
# Summary for numeric columns only
print("=== Numeric Summary ===")
print(df.describe())

# Summary for categorical columns only
print("\n=== Categorical Summary ===")
print(df.describe(include=['object']))


This way, you don’t mix stats like "mean" with "most frequent label".

🔹 Option 2: Use include='all'
print("=== Full Summary (numeric + categorical) ===")
print(df.describe(include='all'))


👉 Downside: lots of NaN values will appear because numeric columns don’t have "unique" or "top", and categorical columns don’t have "mean" or "std".

🔹 Option 3: Go more detailed with profiling tools

For a richer summary:

pip install ydata-profiling

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Census Data EDA", explorative=True)
profile.to_notebook_iframe()   # if using Jupyter Notebook
# profile.to_file("census_report.html")   # export as full HTML report


This gives per-column statistics, correlations, distributions, and warnings automatically.

✅ For your case (Census dataset with mixed data), the best practice is Option 1 (separate summaries) so that:

Numeric variables get proper statistical measures.

Categorical variables get frequency-based measures.

Do you want me to update your earlier exploration script so it prints numeric and categorical summaries separately (instead of include='all')?
```

- Prompt 4: We are now cleaning the dataset. Generate code that handles missing values and inconsistencies, standardize formats, remove duplicates, detect and treat outliers. Create a new CSV file for the cleaned dataset.

Generated Code:
```
import pandas as pd
import numpy as np

# Column names
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load dataset with missing values already set as NaN
df = pd.read_csv(
    "raw_dataset.csv",
    header=None,
    names=columns,
    na_values="?"
)

# ============================================================
# 1. Handle Missing Values
# ============================================================

# Drop rows where target 'income' is missing (if any)
df = df.dropna(subset=["income"])

# For categorical columns: fill NaN with mode (most frequent value)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# For numeric columns: fill NaN with median (robust to outliers)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# ============================================================
# 2. Standardize Formats
# ============================================================

# Strip extra spaces and lowercase for categorical consistency
for col in categorical_cols:
    df[col] = df[col].str.strip().str.lower()

# Standardize target variable to binary format (<=50k, >50k)
df["income"] = df["income"].replace({"<=50k": "0", ">50k": "1"})

# ============================================================
# 3. Remove Duplicates
# ============================================================
df = df.drop_duplicates()

# ============================================================
# 4. Detect and Treat Outliers (IQR method for numeric columns)
# ============================================================

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of dropping (winsorization)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# ============================================================
# 5. Save Cleaned Dataset
# ============================================================
df.to_csv("cleaned_dataset.csv", index=False)

print("✅ Dataset cleaned and saved as 'cleaned_dataset.csv'")

```


- Prompt 5: Why do I have to 
if set len(mode_values) > 1: mode = np.array(mode_values) 
else: mode = mode_values[0] median = census_df[col].median()?

Generated Code:
```
Ah, nice catch 👍 — you’re bumping into a subtlety with **`mode()` in Pandas**.

---

### 🔎 What’s happening:

* When you do:


  mode_values = census_df[col].mode()


  Pandas returns a **Series**, not a single value.

  * If there’s **one mode** → it looks like `[ 'private' ]`.
  * If there are **multiple modes (ties)** → you’ll get multiple values, like `[ 'private', 'self-emp-not-inc' ]`.

That’s why you sometimes see code like:


if len(mode_values) > 1:
    mode = np.array(mode_values)   # multiple modes
else:
    mode = mode_values[0]          # single mode


---

### 💡 But in practice:

When filling missing values, you don’t usually want *multiple values*. You just need **one representative value**.
Most people just take the **first mode**:


mode = census_df[col].mode()[0]
census_df[col].fillna(mode, inplace=True)


This avoids the whole `if len(mode_values) > 1` check, because `[0]` always selects the first.

---

### 🔹 Why `median` is mentioned:

For **numerical columns**, instead of mode, you often use **median**:


median = census_df[col].median()
census_df[col].fillna(median, inplace=True)


* Median is **robust against outliers** (unlike mean).
* Mode is more for **categorical values**.

---

✅ So, to summarize:
You don’t *have to* use that `if len(mode_values) > 1:` check. It’s just one way of handling multiple modes.
In most cleaning workflows, we just do:

* **Categorical columns** → `df[col].mode()[0]`
* **Numeric columns** → `df[col].median()`

---

Do you want me to **revise your cleaning code** to use the simpler `df[col].mode()[0]` approach instead of that `if...else` check?

```

- Prompt 6: Should I fill the empty non-numerical attributes with the mode value for that attribute? Wouldn't it be ideal to just drop that row completely?

Generated code:

```
Great question — and you’re thinking like a data scientist now ⚡.
The choice between **filling missing categorical values with the mode** vs. **dropping rows entirely** depends on context:

---

### 🔹 Option 1: Fill with Mode (most frequent value)

* ✅ **Keeps all rows** → no data loss.
* ✅ Good if missing values are **relatively few and random** (Missing Completely at Random).
* ❌ Introduces bias → over-represents the most frequent category.
* ❌ May distort relationships if many values are missing.

👉 Best when:

* You don’t want to shrink the dataset.
* Missing values are small (<5–10% of that column).
* The attribute is **important** (dropping rows might delete useful target labels).

---

### 🔹 Option 2: Drop Rows with Missing Categorical Values

* ✅ Avoids bias — no guessing.
* ✅ Cleaner dataset.
* ❌ Reduces dataset size.
* ❌ Dangerous if many rows have missing values (can cause underfitting).

👉 Best when:

* The dataset is **large enough** (like Census data).
* The percentage of missing rows is **tiny** (<1–2%).
* The missingness is **not random** (e.g., “workclass” missing only for certain groups → imputation might hide that pattern).

---

### 🔹 Option 3: Smarter Imputation

Instead of always taking mode, you could:

* Use **“unknown”** or **“missing”** as a category (preserves info that it was missing).
* Use **predictive imputation** (e.g., train a model to guess missing values).

---

### ⚡ For the Census dataset (like UCI Adult Data)

* Missing values are usually in `workclass`, `occupation`, and `native-country`.
* They’re not too many (~5–6%).
* Most studies either:

  1. Replace with `"unknown"` (treat as its own category), OR
  2. Fill with mode (simple, fast), OR
  3. Drop rows (if doing a quick clean, since dataset is big enough).

---

✅ **Practical advice:**

* If you’re exploring or doing ML quickly → **fill with mode or `"unknown"`**.
* If you’re aiming for research-quality cleaning → **check % missing per column**, and drop only if it’s small.

---

Would you like me to **modify your cleaning code** so that instead of mode-filling, we:

1. **Drop rows** if a non-numerical value is missing, OR
2. **Create a new category `"unknown"`** for missing categorical values?
```

- Prompt 7: Explain how we treated the outliers in this part: 
for col in numeric_cols: Q1 = census_df[col].quantile(0.25) Q3 = census_df[col].quantile(0.75) IQR = Q3 - Q1 lower_bound = Q1 - 1.5 * IQR upper_bound = Q3 + 1.5 * IQR # Cap outliers instead of dropping (winsorization) census_df[col] = np.where(census_df[col] < lower_bound, lower_bound, census_df[col]) census_df[col] = np.where(census_df[col] > upper_bound, upper_bound, 
census_df[col])

Generated Answer:
```
This block of code is using a **classical IQR (Interquartile Range) method** for outlier detection, and instead of deleting rows, it **treats outliers by capping them** — a process called **winsorization**. Let’s break it down step by step:

---

### 🔹 Step 1: Calculate the IQR


Q1 = census_df[col].quantile(0.25)   # 25th percentile
Q3 = census_df[col].quantile(0.75)   # 75th percentile
IQR = Q3 - Q1


* `Q1` is the first quartile (25% of data is below this value).
* `Q3` is the third quartile (75% of data is below this value).
* `IQR` is the range where the **middle 50% of the data** lies.

---

### 🔹 Step 2: Define "outlier" thresholds


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


* Anything **below `lower_bound`** or **above `upper_bound`** is considered an **outlier**.
* The `1.5 * IQR` is the conventional Tukey’s rule.

---

### 🔹 Step 3: Treat outliers (Winsorization)

census_df[col] = np.where(census_df[col] < lower_bound, lower_bound, census_df[col])
census_df[col] = np.where(census_df[col] > upper_bound, upper_bound, census_df[col])


* Instead of **dropping** rows with outliers, we **cap** them:

  * Any value smaller than `lower_bound` is set equal to `lower_bound`.
  * Any value larger than `upper_bound` is set equal to `upper_bound`.

This way:

* ✅ Extreme values don’t distort the distribution (e.g., very high income or age).
* ✅ You don’t lose data (rows are preserved).
* ❌ But you lose the original “extremeness” of those values.

---

### 📊 Example:

Suppose `hours-per-week` has most values between 35 and 45, but a few people reported **1 hour** or **99 hours**.

* Q1 = 35, Q3 = 45 → IQR = 10.
* Lower bound = 20, Upper bound = 60.
* Any value <20 is capped to 20, any value >60 is capped to 60.

So:

* `1` becomes `20`.
* `99` becomes `60`.

---

✅ **Summary of what we did**:
We **detected outliers using the IQR rule** and **treated them by capping extreme values** (winsorization) instead of dropping rows. This preserves dataset size while reducing the influence of extreme values on models and statistics.

---

Do you want me to also show you **before vs. after boxplots** for one column (like `hours-per-week`) so you can **visualize how winsorization changed the distribution**?

```

## Results & Snapshots
- Rows before: 32561
- Rows after: 32537

**Before**
1. Sample Rows
<img width="745" height="636" alt="Screenshot 2025-09-30 at 12 03 00 AM" src="https://github.com/user-attachments/assets/cdafd862-380c-41ae-b8ad-669d9b2d971b" />

2. Shape and Non-numerical attributes summary
<img width="745" height="627" alt="Screenshot 2025-09-30 at 12 04 02 AM" src="https://github.com/user-attachments/assets/fb2b80d5-9335-46a3-a91b-9d4f75db9cb7" />

3. Numerical attributes Statistical Summary
<img width="745" height="180" alt="Screenshot 2025-09-30 at 12 05 07 AM" src="https://github.com/user-attachments/assets/ba4d4287-7cd0-4895-a228-fd390bb8a83c" />

**After**
1. Sample Rows
<img width="745" height="488" alt="Screenshot 2025-09-30 at 12 06 07 AM" src="https://github.com/user-attachments/assets/4a2e058c-f0f9-4f33-86bf-b96f5b7e6f3e" />

2. Shape and Non-numerical attributes summary
<img width="745" height="595" alt="Screenshot 2025-09-30 at 12 06 30 AM" src="https://github.com/user-attachments/assets/0ba8aa4d-51a4-4784-891a-fe2a915f92ea" />


3. Numerical attributes Statistical Summary
<img width="745" height="183" alt="Screenshot 2025-09-30 at 12 06 54 AM" src="https://github.com/user-attachments/assets/275bcd28-0b19-4a43-a30d-43d3644a004f" />


