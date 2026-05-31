# Data Science Internship
# **Week 1 Task
A thorough foundational set of tasks encompassing fundamental data engineering and mathematical concepts needed for deep machine learning implementations may be found in this repository. In order to concentrate on the underlying structural code mechanics, each section stays away from opaque black-box framework logic.

## 📁 Repository Structure
Math assertions, structural visualizations, and full code implementations are all contained in the core Google Colab workspace pipeline.

## 🛠️ Detailed Implementations
### 1. Fundamentals of Python and Defensive Programming
* **Control Flow Automation:** Clean, boundary-validated conditional pathways were used to formulate numerical evaluation algorithms.
* **Algorithmic Optimization:** Advanced list comprehensions and sets were used for optimized memory operations, while dynamic string frequencies were processed manually to avoid standard library structures.
* **Exception Layer Design:** Robust, runtime-safe division handlers were created to raise explicit feedback alerts (`TypeError`) when non-numeric parameters are caught and to securely intercept arithmetic anomalies (`ZeroDivisionError`).

### 2. Using NumPy for Scientific Computing
* **Multi-Dimensional Transformations:** From initial 1D vectors to highly targeted 3D multi-layered matrices (`2x2x3`), dimensional mutations were performed across structural representations.
* **Matrix Operations & Invariance Checking:** Element-wise array operations and conventional inner dot-products were assessed, and non-commutative spatial behavior ($P \times Q \neq Q \times P$) was explicitly demonstrated.

### 3. Pandas Data Wrangling and Feature Profiling
* **Relational Query Slicing:** Absolute indexing techniques (`.loc` vs. `.iloc`) based on composite masking operations were used to isolate important target parameters.
* **Production Preprocessing Pipelines:** Developed full missing value recovery tracks using dynamic localized statistics (mean-shift standard approximations for age metrics, and median value column imputations for target outliers).

### 4. Dimensionality Reduction and Linear Algebra
* **Euclidean Norm Calculations:** Used spatial quiver visualizations to compute matrix $L_2$ characteristics and visualize continuous directional coordinates.
* **Spectral Analysis (Eigenpair Extraction):** linear configurations that have been broken down to produce basic transformation attributes (Eigenvalues & Eigenvectors). created assertion layers that confirm the basic identity vector relation:
$$A\vec{v} = \lambda\vec{v}$$ Singular Value Decomposition was used in Data Matrix Compression (SVD) to separate arbitrary observation profiles into primary coordinates ($U, \Sigma, V^T$). To mimic structural PCA patterns, low-rank approximations were performed utilizing leading dominating single elements.

### 5. Exploration and Descriptive Statistics
* **Distribution Characterization:** Complete numerical profiles (Mean, Median, Standard Deviation, and Interquartile Ranges) were computed for continuous fields in order to characterize the distribution.
* Visualizations of Density: To map production variances versus typical normal profiles, kernel density estimates (KDE) were plotted over frequency histograms.

  
---

## 💻 Technical Setup & Stack
* **Language Environment:** Python 3 (Google Colab Environment)
* **Core Packages:** `numpy`, `pandas`, `scipy.stats`, `matplotlib`, `seaborn`

# Week 2 Task

## 📌 Project Overview

This project designs and implements an end-to-end Machine Learning and Statistical Time Series pipeline to predict **Tesla's Monthly Global Vehicle Deliveries**. Utilizing real-world operational tracking datasets spanning from 2015 through 2025, the workflow evaluates predictive capabilities under strict out-of-sample chronological forecasting boundaries.

The primary engineering focus of this project was the structural refactoring of an initial high-performing regression model to eliminate a severe case of **Target Leakage / Look-Ahead Bias**, ensuring true predictive utility in real-world deployment scenarios.

---

## 🛠️ Tech Stack & Dependencies

* **Data Analysis & Profiling:** `Pandas`, `NumPy`
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning Models:** `Scikit-Learn` (Gradient Boosting Regressor, Random Forest Regressor)
* **Statistical Modeling:** `Statsmodels` (Holt-Winters Exponential Smoothing)
* **Data Ingestion:** `KaggleHub`

---

## 🚀 Pipeline Architecture

### 1. Ingestion & Preprocessing

* Automated raw file downloads directly through `kagglehub`.
* Cleaned whitespace inconsistencies in structural column headers.
* Converted fractional time metrics (`Year`, `Month`) into a standardized `DatetimeIndex`.
* Aggregated granular, regional, and model-specific entries to compute continuous **Unified Global Monthly Totals**.

### 2. Exploratory Data Analysis (EDA)

* Profiled selling price distributions using Kernel Density Estimate (KDE) histograms.
* Analyzed total production volume distribution trends categorized by specific vehicle classes.
* Generated a numeric metric correlation matrix to examine multi-collinear interactions between production output and consumer deliveries.

### 3. Feature Engineering & Target Transformation

To construct a valid forecasting environment, the feature matrix was updated using two major techniques:

* **Operational Feature Lagging:** Key predictors (`Production_Units`, `Avg_Price_USD`) were lagged by 1 month ($\text{Lag}_1$), ensuring future target points are calculated exclusively using verified historical context.
* **Stationary Target Differencing:** To overcome the mathematical limitation where tree-based algorithms struggle to extrapolate outside their historical absolute training scale limits, the target metric was transformed into a **Month-over-Month Change** series:

$$\Delta Y_t = Y_t - Y_{t-1}$$



### 4. Validation Framework

* **Chronological Splitting:** Avoided standard randomized shuffling (`train_test_split`) which breaks sequential dependencies.
* **Holdout Test Set:** Segmented the final 12 unique months (**2025-01 to 2025-12**) as an unseen holdout window.
* **Cross-Validation:** Applied `TimeSeriesSplit` cross-validation during the optimization loops to secure parameters without look-ahead contamination.

---

## 🚨 Resolving the Target Leakage Trap

During initial baseline evaluations, standard regression setups yielded an artificial **$R^2$ Score of 0.86 to 0.95**. Technical audit revealed that this high performance was a structural flaw caused by target leakage:

1. Concurrent production units for the same target month were available in the feature matrix.
2. Because Tesla manufactures vehicles to fulfill active order queues, current production has a **0.99 correlation** with current deliveries.

### The Solution:

By explicitly dropping all current-month indicators from the input matrix, shifting operational metrics back by 1 month, and training a **Tuned Gradient Boosting Regressor** exclusively on differenced variations, the look-ahead bias was entirely eliminated. The predicted changes were reconstructed back into true absolute figures during post-processing using the preceding month's actual baseline:


$$\hat{Y}_{\text{Absolute}} = Y_{t-1} + \hat{\Delta Y}_t$$

---

## 📊 Final Performance Scoreboard (2025 Validation Horizon)

| Model Strategy | RMSE | MAE | $R^2$ Score |
| --- | --- | --- | --- |
| **Tuned Gradient Boosting (Stationary Pipeline)** | 14,865.62 | 13,039.42 | **-0.34** |
| **Holt-Winters Baseline (Time Series)** | 11,741.93 | 9,483.22 | **0.17** |

### Key Strategic Insights:

* **Statistical Resilience:** The classical Holt-Winters model achieved a positive $R^2$ score of **0.17**, outperforming machine learning models by cutting through month-over-month noise to model Tesla's long-term 12-month cyclical corporate delivery loops.
* **Volatility Limits:** Pure history-based autoregressive lags cannot anticipate sudden real-world global market adjustments (such as the sharp V-shaped plunge observed in May 2025) without external leading indicators.

---

## 🔮 Future Enhancements (Phase 2 Roadmap)

1. **Exogenous Feature Integration:** Incorporate global macroeconomic signals, including consumer price indices (CPI), electric vehicle battery raw material costs, and regional interest rate updates.
2. **Hybrid Model Architectures:** Implement deep learning sequence models (`LSTM`, `GRU`) paired with linear trend-residual structural decompositions.
