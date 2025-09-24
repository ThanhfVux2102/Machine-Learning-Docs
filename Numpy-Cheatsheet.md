# 🧮 NumPy & Random Cheatsheet

## CREATE

```python
# 0D
arr = np.array(1)

# 1D
ar = np.array([1,2,3,4,5])

# 2D
array2D = np.array([[1,2,3],[4,5,6]])
arr2D = np.array([[1,2,3],[4,5,6],[6,7,8]])

# 3D
arr3D = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])

print(arr3D.ndim)
print(arr3D.shape)
```

---

## INDEXING

```python
# 1D
print(ar[0] + ar[1])

# 2D
print(arr2D[0,1])   # 1st element at 2nd row, 1st col

# 3D
print(arr3D[0,0,1]) # 1st element at 2nd row, 1st col, 1st array
```

---

## SLICING

- **1D**
```python
print(ar[1:5:2])
print(ar[:4:1])
print(ar[-3:-1])  # 3,4
```

- **2D**
```python
print(arr2D[2,1:3])
```

---

## DATA TYPES

- `i` → integer  
- `b` → boolean  
- `u` → unsigned integer  
- `f` → float  
- `c` → complex  
- `m` → timedelta  
- `M` → datetime  
- `O` → object  
- `S` → string  
- `U` → unicode  
- `V` → fixed chunk of memory (void)  

---

## COPY & VIEW

```python
# Copy: không bị ảnh hưởng
x = ar.copy()

# View: bị ảnh hưởng theo array gốc
x = ar.view()
```

---

## SHAPE

```python
arr2 = np.array([[1,2,3,4],[5,6,7,8]])
print(arr2.shape)
```

---

## RESHAPING

```python
# 1D → 2D
arr3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr = arr3.reshape(4,3)

# 1D → 3D
newarr2 = arr3.reshape(2,3,2)
```

---

## LOOP

```python
# 1D
for x in arr3: print(x)

# 2D
for x in arr2: print(x)

# 3D
for x in arr3D: print(x)
```

---

## JOINING ARRAYS

- **Concatenate (1D)**  
```python
a1 = np.array([1,2,3])
a2 = np.array([4,5,6])
a3 = np.concatenate((a1,a2)) # [1,2,3,4,5,6]
```

- **Stack (2D)**  
```python
a4 = np.stack((a1,a2), axis=0)
```
- `axis=0`: theo cột  
- `axis=1`: theo hàng  

---

## SPLITTING

```python
# 1D
newarr3 = np.array_split(ar, 3)

# 2D
newarr4 = np.array_split(arr2D, 3)
```

---

# 🎲 RANDOM : DATA SCIENCE

## DISTRIBUTION

```python
# 1D
x = random.choice([1,3,5,7,9], p=[0.2,0.2,0.2,0.2,0.2], size=100)

# 2D
x = random.choice([2,4,6,8], p=[0.1,0.2,0.5,0.2], size=(2,4))
```

- `p` → probability  
- `size` → shape of array  

---

## PERMUTATIONS

- **Shuffle (in-place)**  
```python
ar2 = np.array([1,2,3,4,5,6,7,8,9])
random.shuffle(ar2)
```

- **Permutation (new array)**  
```python
random.permutation(ar2)
```

---

## SEABORN

```python
sns.displot([0,1,2,3,4,5,6,7,8,9], kind='ecdf')
# kind='kde'  → curve distribution
# kind='hist' → histogram
# kind='ecdf' → cumulative distribution
plt.show()
```

---

## NORMAL DISTRIBUTION

```python
x = random.normal(loc=1, scale=2, size=(2,3))

sns.displot(random.normal(loc=10, scale=5, size=3), kind="kde")
plt.show()
```

- `loc`: mean  
- `scale`: std dev  
- `size`: shape  

---

## BINOMIAL DISTRIBUTION

```python
y = random.binomial(n=10, p=0.5, size=(2,4))
```

- `n`: number of trials  
- `p`: probability per trial  
- `size`: shape  

---

## POISSON DISTRIBUTION

```python
z = random.poisson(lam=2, size=(2,4))
```

- `lam`: rate (# occurrences)  

---

## UNIFORM DISTRIBUTION

```python
xt = random.uniform(low=0.0, high=1.0, size=(2,3))
```

---

# 🔧 NUMPY UFUNC

## Intro

- Tương đương hàm `def` nhưng tối ưu cho vectorization.  
- Dùng `frompyfunc(func, inputs, outputs)`.  

```python
def add(x,y): return x+y
add = np.frompyfunc(add, 2, 1)
print(add([1,2,3,4],[5,6,7,8]))
```

---

## BASIC ARITHMETIC

- `np.add()`  
- `np.subtract()`  
- `np.multiply()`  
- `np.divide()`  
- `np.power()`  
- `np.remainder(x,y)` → sign theo divisor  
- `np.mod(x,y)` → sign theo dividend  
- `np.absolute()`  

---

## ROUNDING DECIMALS

- `np.trunc()` → cắt thập phân  
- `np.fix()` → cắt thập phân  
- `np.round()` → làm tròn chuẩn  
- `np.floor()` → làm tròn xuống  
- `np.ceil()` → làm tròn lên  

---

## SUMMATIONS

```python
a1 = np.array([1,2,3])
a2 = np.array([4,5,6])

np.sum([a1,a2])             # 21
np.sum([a1,a2], axis=1)     # [6,15]
np.sum([a1,a2], axis=0)     # [5,7,9]
np.cumsum(np.array([1,2,3])) # [1,3,6]
```

---

## PRODUCTS

```python
a5 = np.array([1,2,3])
np.prod(a5)         # 6
np.prod([a1,a2], axis=1) # [6,120]
np.cumprod(a5)      # [1,2,6]
```
