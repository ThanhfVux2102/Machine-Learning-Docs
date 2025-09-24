# 📊 Matplotlib Cheatsheet

## PLOTTING

### goals : this function for drawing points (markers)

```python
xpoints = np.array([0,6])
ypoints = np.array([0,10])

plt.plot(xpoints, ypoints)
plt.show()
```

### Plotting without line
```python
plt.plot(xpoints, ypoints, marker='o')
plt.show()
```

### Multiple point
```python
Xpoints = np.array([1,2,3,4])
Ypoints = np.array([5,6,7,8])

plt.plot(Xpoints, Ypoints)
plt.show()
```

### Default X-Points
- if 1 argument → default y  
- if 2 arguments → (x, y)  

```python
ypoint = np.array([1,2,3,4])
plt.plot(ypoint)
plt.show()
```

---

## MARKER

- `'o'` → Circle (hình tròn)  
- `'*'` → Star (ngôi sao)  
- `'.'` → Point (điểm nhỏ)  
- `','` → Pixel (điểm pixel cực nhỏ)  
- `'x'` → X (dấu X rỗng)  
- `'X'` → X filled (dấu X tô đầy)  
- `'+'` → Plus (dấu cộng rỗng)  
- `'P'` → Plus filled (dấu cộng tô đầy)  
- `'s'` → Square (hình vuông)  
- `'D'` → Diamond (hình thoi)  
- `'d'` → Diamond thin (hình thoi mảnh)  
- `'p'` → Pentagon (ngũ giác)  
- `'H'` → Hexagon (lục giác lớn)  
- `'h'` → Hexagon (lục giác nhỏ)  
- `'v'` → Triangle Down (tam giác hướng xuống)  
- `'^'` → Triangle Up (tam giác hướng lên)  
- `'<'` → Triangle Left (tam giác hướng trái)  
- `'>'` → Triangle Right (tam giác hướng phải)  

---

## PLT LINE STYLE

```python
plt.plot(ypoint, marker='o', linestyle="--")
plt.show()
```

- `'-'` → Solid line (đường liền)  
- `'--'` → Dashed line (đường gạch ngang)  
- `'-.`' → Dash-dot line (đường gạch + chấm)  
- `':'` → Dotted line (đường chấm)  
- `''` hoặc `' '` → No line (chỉ vẽ marker)  

---

## LABEL

```python
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot Title')
```

---

## GRID

- Add grid to all plot:  
  ```python
  plt.grid()
  ```

- Add grid for x-axis:  
  ```python
  plt.grid(axis='x')
  ```

- Add grid for y-axis:  
  ```python
  plt.grid(axis='y')
  ```

---

## SUPLOT

- **Syntax**: `plt.subplot(x, y, z)`  
  - `x`: số hàng  
  - `y`: số cột  
  - `z`: vị trí biểu đồ  

Ví dụ:  
```python
plt.subplot(1,2,1)
```
→ tạo 2 biểu đồ cạnh nhau theo chiều ngang.  

---

## SCATTER : phân tán

```python
x = np.array([1,2,3,43,54,69,73,8])
y = np.array([12,99,14,15,16,8,120,23])

plt.scatter(x, y)
plt.show()
```

- Thay đổi màu:
```python
plt.scatter(x, y, color='#88c999')
plt.show()
```

---

## BAR

### Vertical Bar
```python
plt.bar(x, y)
plt.show()
```

### Horizontal Bar
```python
plt.barh(x, y)
plt.show()
```

- **Parameters**:  
  - `width` → chỉ áp dụng cho horizontal  
  - `height` → chỉ áp dụng cho vertical  

---

## HISTOGRAM

```python
x = np.array.random.normal(loc=40, scale=10, size=100)
plt.hist(x)
plt.show()
```

---

## PIE CHART

```python
y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(y, labels=mylabels)
plt.show()
```
