### Description

This repository contains the code for a Gaussian Process Regression (GPR) framework with embedded feature selection for remote sensing estimation of chlorophyll-*a* (Chl*a*) concentration in coastal waters. The framework integrates Automatic Relevance Determination (ARD) with two feature selection strategies — the Derivative Decomposition Ratio (DDR) and Sensitivity Analysis (SA) — enabling systematic identification of optimal spectral predictors across diverse optical water types (OWTs) and multiple satellite sensors.

---

#### Background

Chl*a* is a key indicator of phytoplankton biomass and marine primary productivity, widely estimated from satellite remote sensing reflectance (*R*rs(λ)). Accurate retrieval in optically complex coastal waters is challenging due to the co-influence of suspended particulate matter and colored dissolved organic matter (CDOM) on the reflectance signal. While machine learning methods — particularly Gaussian Process Regression — have demonstrated strong performance for Chl*a* retrieval, effective feature selection from high-dimensional spectral inputs remains a critical and unresolved challenge. This repository addresses that gap by providing two GP-embedded feature selection approaches and evaluating their comparative performance.

---

#### Dataset

The framework was developed using a globally representative *in-situ* dataset of 6,015 quality-controlled coastal samples aggregated from five publicly available databases:

- **GBIDO** – A Compilation of Global Bio-Optical In Situ Data for Ocean-Colour Satellite Applications v3 (Valente et al., 2022)
- **GLORIA** – A Globally Representative Hyperspectral In-Situ Dataset for Optical Sensing of Water Quality (Lehmann et al., 2023)
- **NWESS** – Northwest European Shelf Seas dataset (Hadjal et al., 2022)
- **CoASTS-BiOMaP** – Coastal Atmosphere and Sea Time Series and Bio-Optical Mapping of Marine Properties (Zibordi & Berthon, 2024)
- **BRAZA** – A Bio-Optical Database for Remote Sensing of Water Quality in Brazilian Coastal and Inland Waters (Maciel et al., 2025)

The combined dataset spans more than three orders of magnitude in Chl*a* (0.03–53.0 mg m⁻³), covering oligotrophic to ultra-eutrophic conditions. *R*rs(λ) spectra were aggregated to the spectral bands of four satellite sensors: **SeaWiFS** (412–670 nm), **MERIS** (412–681 nm), **MODIS** (412–678 nm), and **VIIRS** (412–672 nm). Independent validation was performed using SeaBASS satellite matchup datasets for all four sensors.

---

#### Optical Water Type Classification

Before model development, the dataset was stratified into five OWTs using unsupervised hierarchical clustering (Ward's method) applied to normalized *R*rs(λ) spectra. Five distinct optical classes were identified, ranging from clear oligotrophic waters (OWT-1, mean Chl*a* = 0.14 mg m⁻³) to extremely turbid, non-algal-particle-dominated environments (OWT-5, mean Chl*a* = 7.56 mg m⁻³). OWT membership for satellite-derived reflectance spectra was assigned probabilistically using the Mahalanobis distance, enabling adaptive model application according to the optical characteristics of each pixel.

---

#### Methodology

**1. GPR with ARD Kernel**

The core regression model is a Gaussian Process with a Squared Exponential kernel incorporating Automatic Relevance Determination (ARD). Each input feature *d* is assigned an individual length-scale parameter *l*d, which governs the sensitivity of the latent function to that dimension. Features with small length-scales exert a stronger influence on model output and are considered more relevant; features with large length-scales contribute negligibly and can be pruned. Kernel hyperparameters — including length-scales, signal variance, and noise variance — are estimated by maximizing the log marginal likelihood via gradient-based optimization (L-BFGS). This probabilistic framework provides both predictive mean and uncertainty estimates, enabling direct assessment of retrieval confidence.

**2. Input Features**

For each sensor, the input feature vector comprises: (i) *R*rs(λ) at individual spectral bands, (ii) all pairwise band ratios *R*rs(λ*i*)/*R*rs(λ*j*), and (iii) four higher-dimensional spectral descriptors (*x*e1–*x*e4) sensitive to chlorophyll absorption and scattering characteristics. This yields between 19 (VIIRS) and 40 (MERIS) candidate input features per sensor. Models are trained and evaluated independently for each sensor–OWT combination.

**3. DDR-Based Feature Selection**

The Derivative Decomposition Ratio (DDR) provides a normalized, sample-wise decomposition of GPR output sensitivity. For each input sample, the squared partial derivative of the predictive mean with respect to each feature is computed and normalized by the sum of squared partial derivatives across all features, yielding a per-sample importance score bounded between 0 and 1. Global feature importance is obtained by averaging DDR scores over all training samples. Features are ranked in descending order of importance, and a cumulative importance threshold (≥0.99) is applied to identify the minimal feature subset that captures the dominant predictive information. The optimal subset corresponds to the first elbow point at which test RMSE stabilizes and R² ceases to show meaningful improvement. DDR explicitly quantifies each feature's proportional contribution to total output variability, overcoming the limitations of conventional ranking-only approaches and enabling systematic, reproducible, and physically interpretable feature subset selection.

**4. SA-Based Feature Selection**

Sensitivity Analysis (SA) within the GPR framework provides a complementary, global measure of input feature relevance derived directly from the ARD kernel hyperparameters. Two sensitivity measures are computed: (i) the sensitivity of the predictive mean (*S*μ), which quantifies the expected influence of each input dimension on model output, and (ii) the sensitivity of the predictive variance (*S*σ), which reflects each feature's contribution to predictive uncertainty. Features are ranked jointly according to both criteria — prioritizing those with high mean sensitivity and low variance sensitivity — to identify spectral bands that are both predictively relevant and statistically stable. The optimal feature subset is determined progressively: starting from the most relevant feature, additional bands are incrementally incorporated and the GPR model is retrained at each step, with the final selection made at the point where performance metrics stabilize. While SA provides a computationally convenient ranking mechanism, it yields relative importance scores rather than normalized contributions, and does not provide an explicit criterion for cumulative subset selection.

**5. Comparison of DDR and SA**

The two feature selection methods differ fundamentally in how they quantify and select features. SA produces global sensitivity scores derived from kernel hyperparameters and the full training dataset, providing a coarse-grained measure of feature relevance that reflects average behavior across the input space. DDR, by contrast, operates at the sample level, decomposing output sensitivity into normalized per-feature contributions for each individual observation. This makes DDR sensitive to local spectral variations and enables it to capture feature interactions that vary across OWTs and optical conditions. In terms of feature set size, SA consistently selects larger subsets (10–17 features), while DDR produces the most parsimonious selections (4–6 features). Despite using fewer inputs, DDR-based GPR models achieve equal or superior predictive performance, particularly in optically complex waters, suggesting that DDR more effectively isolates the dominant spectral predictors while suppressing redundant and correlated inputs.

---

#### Model Evaluation

Model performance is assessed using R², RMSE, MAE, MAPE, and Symmetric Signed Percentage Bias (SSPB). Robustness is evaluated through 101 bootstrap iterations of 10-fold cross-validation, with the Robustness Ratio (RR) — defined as the ratio of cross-validation R² to goodness-of-fit R² — used to quantify the degree of overfitting. DDR-based GPR achieves overall R² = 0.92–0.94 and MAPE = 16.59–18.48% across all sensors on *in-situ* test data, and substantially outperforms standard NASA chlor-a products in independent satellite matchup validation, with MAPE reductions exceeding 50% for SeaWiFS and MODIS. Spatial mapping in the English Channel and Bay of Rio further confirms the model's capacity to reproduce large-scale and mesoscale Chl*a* gradients across contrasting optical environments.

---

#### Reference

If you use this code, please cite:

> Moradi, M., Lu, M., & Arabi, B. (*in preparation*). Gaussian Process Embedded Feature Selection based on Automatic Relevance Determination for Remote Sensing Estimation of Chlorophyll-*a* in Coastal Waters.

---

This description is structured to serve directly as your GitHub README. Let me know if you'd like to adjust the tone, add installation/usage instructions, or include badges (license, Python version, DOI, etc.).
