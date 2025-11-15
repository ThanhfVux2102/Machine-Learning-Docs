# 📊 Visualization Types, Goals, and Suitable Preprocessing Methods

This guide connects each **visualization type** with its **main analytical goal** and the **preprocessing methods** that best support it.  
Useful for EDA (Exploratory Data Analysis) and machine learning preprocessing.

---

| **Visualization Type**                          | **Goal / Purpose**                                   | **Recommended Preprocessing Methods**                                                                                                                                       |
| ------------------------------------------------ | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Histogram**                                   | View the **distribution of a single feature**        | - If distribution is **skewed**: apply **log transform**, **Box-Cox**, or **StandardScaler**  <br> - If there are **outliers**: consider **clipping** or **RobustScaler**    |
| **Box Plot**                                   | Detect **outliers**, compare groups                  | - Handle outliers (via **IQR** or **Z-score**)  <br> - Choose **RobustScaler** instead of MinMax if outliers exist                                                          |
| **Scatter Plot / Pair Plot**                   | Observe **relationships between two variables**      | - Remove or reduce features with **high correlation** (to avoid multicollinearity) <br> - Detect **non-linear relationships** → perform **feature engineering** accordingly  |
| **Heatmap (Correlation Matrix)**                | Visualize **feature correlation strength**           | - Reduce **multicollinearity** by removing redundant features  <br> - Identify **important predictors** for the target variable                                              |
| **Count Plot / Bar Plot**                       | View **frequency of categorical variables**          | - Detect **class imbalance**  <br> - Use **SMOTE**, **undersampling**, or **class_weight** to balance the dataset                                                           |
| **Missing Value Map (sns.heatmap(df.isnull()))** | Locate **missing data**                              | - Decide whether to **fillna()**, **impute** with mean/median/mode, or **drop columns/rows**                                                                                |
| **Line Plot / Trend Plot (Time series)**         | Analyze **trend over time**                          | - Apply **time-series preprocessing** like `rolling mean`, `diff()`, or **seasonal decomposition**                                                                          |

---

## 💡 Quick Reference Summary

| **Question You’re Asking** | **Use This Visualization** | **Common Preprocessing** |
|-----------------------------|-----------------------------|----------------------------|
| What is the **distribution** of values? | Histogram / KDE / Boxplot | Scaling, log-transform, handle outliers |
| How do **groups differ**? | Boxplot / Violin / Barplot | Encoding, outlier detection |
| Are features **correlated**? | Scatter / Pairplot / Heatmap | Correlation filtering, feature selection |
| How do **values change over time**? | Line / Area chart | Rolling mean, differencing, resampling |
| What’s the **balance between categories**? | Countplot / Pie chart | SMOTE, oversampling, class weighting |
| Where are **missing values**? | Missing Value Map | Imputation or dropping |

---



