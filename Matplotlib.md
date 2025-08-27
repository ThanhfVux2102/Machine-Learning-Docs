# 📚 Matplotlib Comprehensive Summary

Organized into two big parts:  
1. **Pyplot (Charts & Graphs)**  
2. **Image (imshow & Image Processing)**  

---

## 🔹 1. Pyplot (Charts & Graphs)

### 1.1 Basic Plot
```python
plt.plot([1,2,3,4], [1,4,9,16], 'ro')

    'r' → red color

    'o' → circle marker

    Format string syntax:

    color + marker + linestyle

Colors

    b → blue

    g → green

    r → red

    c → cyan

    m → magenta

    y → yellow

    k → black

    w → white

Markers

    o → circle

    s → square

    ^ → triangle up

    v → triangle down

    > → triangle right

    < → triangle left

    . → point

    , → pixel

    + → plus

    x → cross

    * → star

    D → diamond

    p → pentagon

    h, H → hexagon

Line Styles

    - → solid

    -- → dashed

    -. → dash-dot

    : → dotted

1.2 Scatter Plot

plt.scatter(x, y, c='red', s=100, marker='o', alpha=0.7)

    c → color

    s → size of marker

    marker → marker style

    alpha → transparency (0 to 1)

With data=...:

plt.scatter('a','b',c='c',s='d',data=data)

    'a' = column for x values

    'b' = column for y values

    'c' = column for colors

    'd' = column for marker sizes

1.3 Subplots

    plt.figure() → create a new figure

    plt.subplot(mnp) → grid layout with m rows, n cols, select panel p

Example:

plt.subplot(211)  # 2 rows, 1 col, top plot
plt.subplot(212)  # bottom plot

1.4 Histogram

n, bins, patches = plt.hist(x, bins=50, density=True, facecolor='g', alpha=0.75)

    n → heights of bars (counts or densities)

    bins → edges of intervals

    patches → list of rectangle objects (each bar)

Types of bins:

    Integer → number of bins (equal width)

    List/array → explicit bin edges

    range(256) → 0–255 bins (common for images)

1.5 Axis & Grid

    plt.axis([xmin, xmax, ymin, ymax]) → set axis limits

    plt.grid(True) → enable grid

1.6 Line2D Object

line, = plt.plot(x, y, '-')
line.set_antialiased(False)

    plt.plot() returns list of Line2D objects

    Unpack the first one: line, = ...

    Methods include:

        .set_color()

        .set_linewidth()

        .set_linestyle()

        .set_antialiased()

1.7 Jupyter/IPython Integration

    %matplotlib inline → static plots in notebook

    %matplotlib notebook → interactive plots (zoom/pan)

    %matplotlib widget → widget-based interactivity

🔹 2. Image (imshow & Processing)
2.1 Displaying Images

imgplot = plt.imshow(img)
plt.show()

    Input: NumPy array

        2D → grayscale

        3D (H×W×3) → RGB

    Returns an AxesImage object (you can call .set_* methods on it).

2.2 Colormaps (cmap)

plt.imshow(img, cmap='gray')

Categories:

    Perceptually Uniform (recommended)

        viridis, plasma, inferno, magma, cividis

    Sequential

        Greys, Blues, Reds, YlGnBu, …

    Diverging

        RdBu, coolwarm, PiYG, BrBG, …

    Qualitative

        Set1, Set2, tab10, Pastel1, …

    Misc (classic)

        jet, hot, cool, spring, summer, autumn, winter

2.3 Histogram of Image Pixels

plt.hist(img.ravel(), bins=256, fc='k', ec='k')

    img.ravel() → flatten array to 1D

    bins=256 → full range for 8-bit images

    Shows brightness/contrast distribution

2.4 Contrast Limits (clim)

plt.imshow(img, clim=(0,175))
# or
imgplot = plt.imshow(img)
imgplot.set_clim(0,175)

    Restrict value range mapped to colormap

    Enhances contrast by focusing on a smaller range

2.5 Interpolation

Defines how pixel values are filled when resizing image.

plt.imshow(img, interpolation='bicubic')

Available methods:

    nearest → blocky, pixelated

    bilinear → smooth (4-neighbor average)

    bicubic → smoother (16-neighbor cubic interpolation)

    lanczos → sharp, high-quality, slower

    Others: spline16, spline36, hamming, hermite, etc.