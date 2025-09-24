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

### Viewing the Data
```python
print(df.head(10))
print(df.info())
```
