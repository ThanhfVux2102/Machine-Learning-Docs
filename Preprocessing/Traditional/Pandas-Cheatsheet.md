# 📊 Pandas Cheatsheet

## Pandas Series: 1-Dimensional Array

### Method 1: `np.linspace(start, stop, num, endpoint=True)`
- **Parameters**:  
  - `start`: first value  
  - `stop`: last value  
  - `num`: number of values  
  - `endpoint`: if `True` (default), includes `stop`; if `False`, excludes `stop`.

```python
seri1 = pd.Series(np.linspace(1,10,5))
print(seri1)
```

### Method 2: `np.random.normal()` and `np.random.rand()`
- `np.random.normal(loc=10, scale=5, size=10)`  
- `np.random.rand(10)`  

```python
seri2 = pd.Series(np.random.normal(loc = 10, scale = 5, size = 10))
seri3 = pd.Series(np.random.rand(10))
sns.displot(seri3, kind = 'kde')
plt.show()
```

---

## Pandas DataFrame: 2-Dimensional Array

**Syntax**:  
```python
pd.DataFrame(data=x, index=y, columns=z, dtype=j, copy=k)
```

- **Parameters**:  
  - `data`: ndarray (structured or homogeneous)
  - `index`: index labels  
  - `columns`: column names  
  - `dtype`: data type  
  - `copy`: whether to copy the data

---

## Pandas Read CSV

**Syntax**:  
```python
df = pd.read_csv('data.csv')
```

### Notion:
- **max rows**: By default, the system has a max number of rows to display. We can change it:
```python
print(pd.options.display.max_rows)  # return 60
pd.options.display.max_rows = 999
print(pd.options.display.max_rows)  # return 999
```

---

## Pandas Read JSON

**Syntax**:  
```python
df = pd.read_json('data.json')
```

### Notion:
- To print the entire DataFrame, use `to_string()`:
```python
print(df.to_string())
```

---

## Analyzing DataFrames


### 👁️ Viewing the Data
```python
print(df.head(10))
print(df.info())
```
| Purpose | Command | Description |
| --- | --- | --- |
| View the first rows | `df.head(10)` | Displays the first 10 rows of the DataFrame |
| General info | `df.info()` | Shows non-null counts, data types, and memory usage |
| Check shape | `df.shape` | Returns (number of rows, number of columns) |
| Check column types | `df.dtypes` | Reveals data types of each column |
| Summary statistics | `df.describe()` | Shows mean, std, min, max, and percentiles |

> 💡 Use this right after loading your dataset to understand its structure and quality.

---

### 🧠 Checking Data Structure
```python
df.shape
df.dtypes
df.info()
df.head()
df.tail()
```
| Purpose | Command | Description |
| --- | --- | --- |
| Check dataset size | `df.shape` | Returns the number of rows and columns |
| Inspect data types | `df.dtypes` | Displays data types of each feature |
| Get quick summary | `df.info()` | Provides overview of non-null values and memory usage |
| Preview top rows | `df.head()` | Shows first few records |
| Preview bottom rows | `df.tail()` | Shows last few records |

---

### 💧 Detecting and Handling Missing / Duplicate Data
```python
df.isnull().sum()
df.dropna()
df.fillna(value)
df.duplicated()
df.drop_duplicates()
```
| Purpose | Command | Description |
| --- | --- | --- |
| Count missing values | `df.isnull().sum()` | Summarizes NaN counts per column |
| Drop missing rows | `df.dropna()` | Removes rows with missing values |
| Fill missing values | `df.fillna(value)` | Replaces NaN with a specific value (e.g., mean or median) |
| Check duplicates | `df.duplicated()` | Returns True for duplicated rows |
| Remove duplicates | `df.drop_duplicates()` | Deletes duplicate records |

> 💡 These are essential steps to clean your data before training.

---

### 📊 Descriptive Statistics & Exploration
```python
df.describe()
df['col'].value_counts()
df.groupby('col').mean()
df.corr()
```
| Purpose | Command | Description |
| --- | --- | --- |
| Summary statistics | `df.describe()` | Count, mean, std, min, max, quartiles |
| Value distribution | `df['col'].value_counts()` | Frequency of unique values |
| Group analysis | `df.groupby('col').mean()` | Computes statistics by group |
| Correlation matrix | `df.corr()` | Measures linear relationships between numeric features |

> 💡 Use this for EDA (Exploratory Data Analysis) to discover relationships and detect imbalance or skewness.

---

### 🧩 Filtering, Selecting, and Transforming Data
```python
df['col']
df[['c1', 'c2']]
df[df['Age'] > 30]
df['BMI'] = df['Weight'] / (df['Height']/100)**2
df.rename(columns={'old':'new'})
df['Gender'].map({'Male':0, 'Female':1})
df['col'].apply(lambda x: x**2)
```
| Purpose | Command | Description |
| --- | --- | --- |
| Select columns | `df['col']`, `df[['c1','c2']]` | Extract one or more columns |
| Filter rows | `df[df['Age'] > 30]` | Filter using conditions |
| Create new column | `df['BMI'] = ...` | Generate a new feature |
| Rename columns | `df.rename(columns={'old':'new'})` | Rename for clarity |
| Map categorical values | `df['Gender'].map({'Male':0, 'Female':1})` | Quick label encoding |
| Apply function | `df['col'].apply(lambda x: x**2)` | Apply transformations |

> 💡 Great for preparing meaningful, well-structured features before feeding into models.

---

### 🧮 Preparing Data for Machine Learning
```python
X = df.drop('target', axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pd.get_dummies(df)
pd.merge(df1, df2, on='id')
```
| Purpose | Command | Description |
| --- | --- | --- |
| Separate features and target | `X = df.drop('target', axis=1)` / `y = df['target']` | Define input (X) and output (y) |
| Split train/test | `train_test_split(X, y)` | Prepare training and evaluation sets |
| Encode categorical variables | `pd.get_dummies(df)` | One-hot encoding |
| Merge DataFrames | `pd.merge(df1, df2, on='id')` | Combine datasets by key |

> 💡 This step bridges data cleaning with model training, making your dataset ready for scikit-learn or deep learning frameworks.

