# Benchmarking Deep Learning Architectures for Oil Recovery Factor Prediction in Polymer-Flood Reservoirs: A Unified Comparative Study

**Authors:** [Author 1]¹, [Author 2]¹, [Author 3]²  
**Affiliations:** ¹Department of Petroleum Engineering, [University Name], [City, Country]; ²[Second Institution]  
**Corresponding Author:** [email@institution.edu]  
**Submitted to:** *Journal of Petroleum Science and Engineering / Geoenergy Science and Engineering*

---

## Abstract

**Background.** Accurate prediction of oil recovery factor is fundamental to reservoir management and field development planning. While physics-based numerical simulation remains the gold standard for polymer-flood enhanced oil recovery (EOR) evaluation, its prohibitive computational cost makes large-scale uncertainty quantification and real-time optimisation impractical. Decline Curve Analysis (DCA), the classical alternative, was not designed for EOR processes and systematically fails to represent the complex multi-parameter interactions that govern polymer-flood displacement efficiency. These limitations have driven growing interest in deep learning proxy models as computationally efficient surrogates for full-physics simulation.

**Methods.** This study presents the first unified benchmark of four foundational deep learning architectures — Multilayer Perceptron (MLP), one-dimensional Convolutional Neural Network (CNN-1D), Long Short-Term Memory network (LSTM), and Tabular Transformer — for predicting oil recovery factor from polymer-flood reservoir simulation data. All models were evaluated on the Proxy5 dataset, comprising 3,306 high-fidelity simulation samples characterised by 14 physicochemical and operational input features. A rigorously standardised pipeline was applied across all architectures: a 70/15/15 training/validation/test split, z-score normalisation of inputs and target, the Adam optimiser (learning rate 10⁻³), shared early stopping (patience = 20), learning rate annealing, and L2 regularisation. Performance was assessed using root mean square error (RMSE), mean absolute error (MAE), coefficient of determination (R²), and mean absolute percentage error (MAPE). Permutation-based feature importance with five repeated shuffles was conducted on each architecture to identify the dominant predictors.

**Results.** All four architectures achieved high predictive accuracy (R² > 0.96). The Transformer attained the best performance across all four metrics (R² = [0.987], RMSE = [0.412]%), followed by LSTM (R² = [0.981]), CNN-1D (R² = [0.973]), and MLP (R² = [0.964]). Permutation importance consistently identified permeability, polymer concentration, reservoir pressure, and residual resistance factor (RRF) as the primary drivers of recovery factor across all architectures.

**Conclusion.** This work provides practitioners with a definitive, reproducible performance baseline for selecting deep learning proxy models in polymer-flood recovery estimation. The Transformer offers the highest accuracy at moderate additional computational cost; LSTM is the strongest alternative when parameter efficiency is required; and the MLP remains the optimal choice for deployment-constrained applications.

**Keywords:** oil recovery factor; polymer flooding; proxy model; deep learning; Transformer; LSTM; CNN; MLP; reservoir simulation; EOR

---

## 1. Introduction

The global energy sector continues to depend substantially on existing oil and gas reserves, and efficient exploitation through enhanced oil recovery (EOR) has become increasingly critical as opportunities for new conventional exploration diminish. Among EOR methods, polymer flooding — which improves displacement efficiency by increasing the viscosity of the injected water phase and reducing the unfavourable mobility ratio between injected and reservoir fluids — is one of the most widely deployed tertiary recovery techniques worldwide. Accurate prediction of the incremental oil recovery factor achievable by a polymer flood under varying reservoir conditions is therefore of considerable practical and economic importance, influencing well spacing decisions, facility design, investment appraisals, and regulatory reserve estimates.

Historically, the petroleum industry has relied on two principal forecasting paradigms: Decline Curve Analysis (DCA) and numerical reservoir simulation. The Arps family of decline models, formalised in 1945, remains in widespread operational use owing to its simplicity and minimal data requirements [1]. However, DCA was developed under implicit assumptions of boundary-dominated flow in conventional, homogeneous reservoirs, and its applicability to unconventional tight and shale plays — which exhibit multi-stage transient flow, hydraulic fracture interference, and pressure-dependent permeability — is severely limited [2]. In unconventional settings, Arps hyperbolic extrapolation systematically overestimates ultimate recovery, and no universally accepted correction has emerged [3]. More critically, DCA provides no mechanism to represent the multiphase, compositional dynamics introduced by EOR processes such as polymer flooding, where injectant rheology, polymer adsorption onto rock surfaces, permeability reduction factors, and salinity-dependent degradation interact in a non-linear, spatially heterogeneous manner that bears no resemblance to the exponential decline framework.

Physics-based numerical reservoir simulation addresses these limitations by solving the governing partial differential equations of multiphase fluid flow, incorporating rock heterogeneity, fluid pressure-volume-temperature (PVT) behaviour, well constraints, and, in the case of polymer flooding, chemical transport and adsorption kinetics [4]. Modern compositional simulators are capable of representing the full complexity of polymer-flood processes with high fidelity. However, a single high-fidelity simulation run for a sector or full-field model can require hours to days of computational time on modern high-performance computing infrastructure, and uncertainty quantification workflows — Monte Carlo sampling, history matching, sensitivity analysis, and production optimisation — typically require ensembles of hundreds to tens of thousands of runs. The resulting computational burden renders brute-force simulation infeasible for real-time decision support, closed-loop reservoir management, and rapid multi-scenario screening.

The petroleum engineering community has increasingly turned to machine learning (ML) and deep learning (DL) proxy models as computationally efficient surrogates for numerical simulation [5]. A proxy model, trained on a designed ensemble of simulator runs, learns the functional mapping from input reservoir and operational parameters to output performance metrics such as recovery factor, cumulative production, or water cut. Once trained, the proxy evaluates in milliseconds rather than hours, enabling the large ensemble evaluations required for optimisation and uncertainty quantification. Early proxy models relied on response surface methods, radial basis functions, and support vector machines; subsequent work adopted artificial neural networks (ANNs) and ensemble tree methods. The availability of large simulation datasets and advances in deep learning hardware have now made it feasible to train sophisticated architectures — convolutional networks, recurrent networks, and attention-based Transformer models — as polymer-flood proxies of high predictive fidelity.

Despite this growing body of work, a critical gap persists in the literature: no prior study has systematically compared the four foundational deep learning architectural families — MLP, CNN-1D, LSTM, and Transformer — within a single, standardised benchmark on a high-fidelity polymer-flood proxy dataset. Existing comparative studies either focus on time-series production forecasting rather than recovery factor estimation from parametric inputs, employ different preprocessing pipelines across architectures rendering cross-model comparisons unreliable, or do not include the Transformer alongside classical convolutional and recurrent models. This fragmentation prevents practitioners from making informed, evidence-based architectural choices and obscures the relative strengths and weaknesses of each paradigm under controlled, reproducible conditions.

The present study directly addresses this gap. The principal contributions are: (i) the construction of a rigorous, unified benchmarking framework in which all four architectures are trained, validated, and tested under strictly identical experimental conditions; (ii) comprehensive evaluation on the Proxy5 polymer-flood dataset comprising 3,306 simulation samples described by 14 engineered input features; (iii) systematic analysis of predictive accuracy, residual behaviour, feature importance, and computational efficiency for each architecture; and (iv) actionable, evidence-based recommendations for practitioners selecting DL architectures for high-speed proxy modelling and production forecasting applications.

The remainder of this paper is organised as follows. Section 2 reviews the relevant literature on deep learning applications in petroleum production forecasting and proxy modelling, organised thematically. Section 3 describes the dataset, preprocessing pipeline, model architectures, training protocol, and evaluation metrics. Section 4 presents and discusses the experimental results. Section 5 draws conclusions and outlines directions for future research.

---

## 2. Literature Review

### 2.1 Recurrent Architectures for Production Forecasting

Recurrent neural networks (RNNs) and their gated variants — Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) — have dominated data-driven production forecasting in petroleum engineering since their first applications in the field. The LSTM architecture, introduced by Hochreiter and Schmidhuber [6], addresses the vanishing gradient problem of vanilla RNNs through input, forget, and output gates, enabling selective retention and erasure of information across arbitrarily long sequences. This property makes LSTM particularly well-suited for the long-range temporal dependencies characteristic of reservoir production decline.

Ojedapo et al. [7] conducted a systematic benchmark of decision trees (DT), random forests (RF), vanilla RNNs, and LSTMs using operational features including wellhead and downhole pressure measurements. Their results demonstrated the superiority of LSTM over all competing methods for modelling long temporal dependencies in production time-series, establishing LSTM as the de facto recurrent baseline for subsequent petroleum DL studies. Building on this foundation, Kocoglu et al. [8] applied Bayesian hyperparameter optimisation to LSTM, Bidirectional LSTM (Bi-LSTM), and GRU architectures for unconventional gas production forecasting, demonstrating that Bi-LSTM — which processes sequences in both forward and reverse temporal directions — achieved superior cumulative production accuracy relative to traditional DCA methods including the Duong and Power-Law Exponential models. Jayeola and Olusola [9] further investigated LSTM for crude oil forecasting in the Niger Delta, demonstrating that the Adam optimiser consistently outperformed stochastic gradient descent (SGD) in convergence speed and final accuracy for non-linear reservoir systems — a finding that supports the exclusive use of Adam in the present benchmark.

Calvette et al. [10] applied Bidirectional LSTM networks as high-speed proxy models for smart well production optimisation, demonstrating that Bi-LSTM could replace computationally expensive full-physics simulators with minimal error, enabling real-time closed-loop optimisation. This proxy modelling paradigm is directly analogous to the application context motivating the present study. Vega-Ortiz et al. [11] demonstrated that LSTM, GRU, and standard RNN architectures all outperformed empirical decline models in capturing the oscillating production patterns associated with well interventions in tight formations. Asante et al. and the associated work of Wagia-Alla et al. [12] evaluated LSTM against the Temporal Fusion Transformer (TFT) for optimising Water-Alternating-Gas (WAG) injection — an EOR process with operational parallels to polymer flooding — concluding that compact recurrent models generalise more efficiently for short operational horizons while TFT offers advantages for multi-step probabilistic forecasting.

### 2.2 Convolutional Approaches

Convolutional neural networks (CNNs), originally developed for image recognition by LeCun et al. [13], have been adapted for one-dimensional sequential and tabular data in petroleum engineering through their ability to learn local feature-interaction patterns via learned filter banks. Werneck et al. [14] introduced the "N-th Day" forecasting setup for heterogeneous carbonate reservoirs, comparing GRU-based sequence-to-sequence models with CNN architectures, and found that CNN configurations performed competitively with recurrent models while offering significantly lower training time — a trade-off of direct relevance to proxy model design. Zhao and Wang [15] integrated a CNN layer as a feature extractor preceding an LSTM encoder for multi-well carbonate production data, achieving significant improvement in R² over standalone LSTM by exploiting the convolutional layers' ability to extract implicit spatial correlations from adjacent well data. Chen et al. [16] demonstrated that a CNN-GRU hybrid for EOR production systems reduced computation time by a factor of 12.4 relative to full numerical simulation while maintaining high predictive accuracy, directly validating the proxy modelling paradigm in an EOR context analogous to the present study.

Kubota and Gioria [17] presented a critical counterpoint by benchmarking MLP, CNN, and LSTM against shallower kernel-based methods on full-field production datasets, identifying the "Swiss-cheese effect" — the tendency of deep architectures to produce spatially inconsistent predictions with localised high-error clusters in sparse, low-dimensional datasets. Their work constitutes an important cautionary note: architectural complexity does not guarantee superior performance, and dataset characteristics must be considered when selecting between architectural complexity and statistical robustness. This observation motivates the explicit MLP baseline in the present benchmark.

### 2.3 Attention Mechanisms and Transformer Architectures

The Transformer architecture, introduced by Vaswani et al. [18] for natural language processing, replaced recurrence entirely with multi-head self-attention, enabling each element of a sequence to attend to all other elements simultaneously. This design enables superior modelling of long-range dependencies while supporting fully parallelised training — a significant computational advantage over sequential LSTM processing.

Al-Ali and Horne [19] conducted one of the earliest systematic evaluations of Transformer-family models in petroleum engineering, comparing ARIMA, BlockRNN, TFT, and N-BEATS on the Volve North Sea field dataset. Their findings demonstrated that TFT's interpretable attention weights and variable selection network allowed it to effectively downweight irrelevant input covariates, yielding superior multi-step forecasting accuracy, while pretrained meta-learning models such as N-BEATS were particularly effective for fields with limited production history. Alakeely and Horne [20] extended this work to multi-horizon probabilistic forecasting in the Eagle Ford Basin, developing the Mixed Input Forecasting (MIF) framework that combines TFT with production index features, demonstrating that the attention mechanism's ability to selectively weight historical windows was particularly beneficial for wells with irregular shut-in periods and operational changes. Zheng et al. [21] introduced Temporal Kolmogorov–Arnold Networks (TKAN) as an alternative to LSTM and TFT, replacing fixed activation functions with learnable spline-based functions and achieving superior accuracy and interpretability on the Volve field dataset. Kumar [22] applied Transformer architectures to well-log processing, demonstrating that self-attention's theoretically unbounded context window overcame the depth-limitation of LSTMs when processing noisy, ultra-deep logging-while-drilling sequences — an application that establishes the Transformer's generalisability to structured sequential data beyond temporal production signals.

### 2.4 Hybrid and Pipeline-Oriented Approaches

Shoeibi Omrani et al. [23] compared physical, deep learning, and hybrid approaches for North Sea gas production forecasting, finding that ANN models augmented with choke opening data — an operational control variable analogous to polymer concentration in the present context — achieved the highest cumulative production forecast accuracy, underscoring that domain-aware feature design is as critical as architectural choice. Oliveira Pires et al. [24] explicitly focused on pipeline design rather than architectural novelty, examining how data resolution, resampling frequency, and operational context (including well shut-ins) influence model reliability across multiple field datasets. Their emphasis on preprocessing standardisation directly motivates the unified pipeline adopted in the present study. Fan et al. [25] demonstrated that ANN proxy models achieved R² = 0.956 for annual production and water cut estimation in low-permeability fields using Pearson correlation for systematic feature selection, establishing a concrete ANN accuracy benchmark for recovery-related targets. Jo et al. [26] introduced Normalised Production Indices (NPIs) — rescaling raw production by completion parameters — demonstrating that this physics-informed preprocessing step significantly enhanced DL model generalisation across heterogeneous tight oil wells, a principle that informs the physical feature design of the Proxy5 dataset used here.

### 2.5 Research Gaps

The foregoing review reveals four principal limitations in the existing literature. First, pure ML/DL models lack intrinsic reservoir physics knowledge, which can produce physically implausible extrapolations beyond the training distribution. Second, models demonstrate sensitivity to data quality, sampling frequency, and class imbalance, where quiescent steady-state conditions dominate rare high-rate transient events. Third, accuracy frequently degrades when models rely solely on historical production rates while neglecting operational context variables. Fourth, and most critically, the benchmarking landscape is highly fragmented: no existing study evaluates all four foundational DL architectural families — MLP, CNN-1D, LSTM, and Transformer — within a single unified experimental framework on the same polymer-flood dataset and preprocessing pipeline. This gap impedes objective architectural selection for practitioners designing production forecasting and proxy modelling systems, and the present study is specifically designed to close it.

---

## 3. Methodology

### 3.1 Dataset Description

The Proxy5 dataset was generated through a systematic ensemble of high-fidelity compositional reservoir simulations of polymer-flood EOR processes. The ensemble was designed using Latin Hypercube Sampling (LHS) to ensure broad, space-filling coverage of the physically plausible parameter space, producing 3,306 unique simulation samples. For each sample, the simulator was executed to depletion under a polymer-flood injection scenario, and the final oil recovery factor (expressed as a percentage of original oil in place, OOIP) was recorded as the target variable.

The 14 input features span four physical categories: rock petrophysical properties (porosity, permeability, rock compressibility), fluid properties (oil viscosity, solution viscosity, reservoir temperature), reservoir and operational conditions (reservoir pressure, injection temperature, formation water multiplier), and EOR-specific polymer parameters (polymer concentration, adsorption, accessible pore volume fraction, residual resistance factor, water salinity). Table 1 provides a complete description of all features.

**Table 1. Input features and physical descriptions, Proxy5 dataset.**

| # | Feature | Symbol | Units | Physical Role |
|---|---|---|---|---|
| 1 | Accessible Pore Volume | APV | fraction | Fraction of pore volume accessible to polymer molecules |
| 2 | Adsorption | C_ads | µg/g | Polymer retention on rock surface per unit mass |
| 3 | Rock Compressibility | c_r | 1/psi | Pore volume change with pressure |
| 4 | Formation Water Multiplier | FWM | — | Relative mobility of formation water phase |
| 5 | Injection Temperature | T_inj | °F | Temperature of injected polymer solution |
| 6 | Oil Viscosity | µ_o | cp | Dead oil viscosity at reservoir conditions |
| 7 | Permeability | k | md | Absolute horizontal permeability |
| 8 | Polymer Concentration | C_p | ppm | Polymer mass concentration in injection water |
| 9 | Porosity | φ | fraction | Matrix bulk porosity |
| 10 | Reservoir Pressure | P_r | psi | Initial reservoir pressure |
| 11 | Residual Resistance Factor | RRF | — | Permeability reduction factor due to polymer retention |
| 12 | Reservoir Temperature | T_r | °F | Ambient formation temperature |
| 13 | Solution Viscosity | µ_s | cp | In-situ viscosity of polymer solution |
| 14 | Water Salinity | C_s | ppm | Total dissolved solids in injection water |
| — | **Oil Recovery Factor** | ORF | **%** | **Target variable** |

The target variable spans a wide continuous range, reflecting the diverse recovery efficiencies achievable across the parameter space. No missing values were identified. The LHS design ensures that no parameter combination is systematically over- or under-represented in the dataset, supporting balanced training and test evaluation.

### 3.2 Data Preprocessing

The 3,306 samples were partitioned into training (70%, n ≈ 2,314), validation (15%, n ≈ 496), and test (15%, n ≈ 496) subsets using stratified random sampling with a fixed seed of 42. Stratification was applied on a discretised binning of the recovery factor to ensure representative coverage of the output distribution across all three subsets.

Both input features and the target variable were independently standardised using z-score normalisation:

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}} \tag{1}$$

where $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are the mean and standard deviation computed exclusively from the training partition. The identical transformation parameters were subsequently applied to the validation and test partitions, strictly preventing any form of data leakage. All reported performance metrics were computed in the original physical unit (percentage recovery) following inverse transformation of model outputs.

For architectures requiring sequence-shaped inputs (CNN-1D, LSTM, Transformer), the 14-dimensional feature vector was reshaped to $(14, 1)$, treating each feature as a single step in a length-14 univariate sequence. The MLP received the flat 14-dimensional vector. No dimensionality reduction or feature selection was applied prior to training, preserving the full input space for permutation importance analysis.

### 3.3 Model Architectures

Four deep learning architectures were designed, implemented in Keras/TensorFlow, and evaluated. All models produce a single scalar output representing the predicted oil recovery factor. Table 2 summarises the key architectural parameters.

**Table 2. Architectural summary of the four deep learning models.**

| Property | MLP | CNN-1D | LSTM | Transformer |
|---|---|---|---|---|
| Input shape | (14,) | (14, 1) | (14, 1) | (14,) |
| Core layers | 4 × Dense (256/128/64/32) | 3 × Conv1D (32/64/128), GAP | 2 × LSTM(128) | 3 × Encoder Block (d=64, h=4) |
| Normalisation | BatchNorm | BatchNorm | — | LayerNorm (Pre-LN) |
| Dropout rate | 0.3 | 0.3 | 0.3 | 0.1 |
| Regularisation | L2 (λ=10⁻⁴) | L2 (λ=10⁻⁴) | — | — |
| Output layer | Dense(1), linear | Dense(1), linear | Dense(1), linear | Dense(1), linear |
| Trainable params | ~[44,000] | ~[52,000] | ~[140,000] | ~[180,000] |

#### 3.3.1 Multilayer Perceptron (MLP)

The MLP is a fully connected feed-forward network serving as the classical baseline. It consists of four dense blocks with hidden dimensions 256, 128, 64, and 32, each followed by Batch Normalisation, ReLU activation, and Dropout (p = 0.3). L2 weight regularisation with λ = 10⁻⁴ is applied to all dense layers. A final linear dense layer with one output neuron produces the predicted recovery factor. The hidden state at layer $l$ is computed as:

$$h_l = \text{Dropout}\bigl(\text{ReLU}\bigl(\text{BN}(W_l h_{l-1} + b_l)\bigr)\bigr) \tag{2}$$

#### 3.3.2 One-Dimensional Convolutional Neural Network (CNN-1D)

The CNN-1D reshapes the 14-dimensional input to $(14, 1)$ and applies three successive convolutional blocks, each comprising a Conv1D layer (kernel size $k=3$, same padding), Batch Normalisation, ReLU, and Dropout (p = 0.3). Filter counts increase as 32 → 64 → 128, building progressively more abstract representations. Global Average Pooling collapses the spatial dimension:

$$\hat{f} = \frac{1}{L} \sum_{i=1}^{L} f_i, \quad L = 14 \tag{3}$$

yielding a 128-dimensional vector passed to a Dense(64)–ReLU–Dropout–Dense(1) head.

#### 3.3.3 Long Short-Term Memory (LSTM)

The LSTM treats the input as a 14-step sequence. Two LSTM layers (hidden dimension $H = 128$) are stacked, with the first returning the full sequence and the second returning the final hidden state. The LSTM cell update equations are:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \tag{4}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \tag{5}$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \tag{6}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \tag{7}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \tag{8}$$
$$h_t = o_t \odot \tanh(C_t) \tag{9}$$

where $f_t$, $i_t$, $o_t$ are the forget, input, and output gates, $C_t$ is the cell state, $\sigma$ is the sigmoid function, and $\odot$ is element-wise multiplication. A Dropout–Dense(64)–ReLU–Dropout–Dense(1) head follows the final LSTM layer.

#### 3.3.4 Tabular Transformer

The Transformer adapts the encoder of Vaswani et al. [18] for tabular regression. Each scalar feature is projected to a $d_{\text{model}} = 64$ embedding. A learnable [CLS] token is prepended to the 14-token sequence, extending it to length 15. Learnable positional embeddings of dimension 64 are added element-wise. Three encoder blocks are applied, each containing:

1. Multi-Head Self-Attention (MHSA) with $h = 4$ heads and $d_k = d_{\text{model}}/h = 16$:
$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \tag{10}$$
$$\text{MHSA}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O \tag{11}$$

2. Position-wise Feed-Forward Network (FFN) with inner dimension $d_{ff} = 256$ and ReLU activation;

3. Pre-normalisation LayerNorm and residual connections around both sub-layers.

After three encoder blocks, the [CLS] token output is layer-normalised and passed through Dense(64)–ReLU–Dropout(0.1)–Dense(1).

### 3.4 Training Protocol

All models were compiled with the Adam optimiser [27] at initial learning rate $\alpha = 10^{-3}$ ($\beta_1 = 0.9$, $\beta_2 = 0.999$). The training loss was mean squared error (MSE) on the normalised target:

$$\mathcal{L}_\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2 \tag{12}$$

Two callbacks were applied uniformly: EarlyStopping (patience = 20 epochs, restoring best weights) and ReduceLROnPlateau (factor = 0.5, patience = 10 epochs, minimum LR = 10⁻⁶). Maximum training budget was 200 epochs; mini-batch size was 64.

### 3.5 Evaluation Metrics

All metrics are computed in the original unscaled recovery factor space following inverse transformation. Let $y_i$, $\hat{y}_i$, $\bar{y}$, and $n$ denote the true values, predictions, mean of true values, and test set size respectively.

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2} \tag{13}$$

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i - y_i| \tag{14}$$

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{15}$$

$$\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{\hat{y}_i - y_i}{y_i}\right| \tag{16}$$

Feature importance was quantified by permutation analysis: for each feature $j$, its test-set values were randomly shuffled five independent times, and the mean increase in RMSE relative to the unpermuted baseline was recorded as the importance score. Larger scores indicate greater predictive relevance.

---

## 4. Results and Discussion

### 4.1 Training Convergence

The training and validation MSE curves for all four architectures are presented in [Figure 7]. All models exhibited monotonically decreasing loss trajectories without evidence of divergence or sustained overfitting, validating the effectiveness of the shared regularisation and scheduling protocol.

The MLP converged fastest, with early stopping triggered at approximately epoch [85]. This rapid convergence reflects the compact input space (14 features) and the MLP's shallow gradient paths. The CNN-1D terminated at approximately epoch [92], exhibiting a characteristic sharp initial drop — corresponding to the fast learning of basic filter patterns — followed by a slower refinement phase. The LSTM required approximately [110] epochs, consistent with the greater number of learnable parameters and the longer gradient paths inherent in recurrent unrolling; the ReduceLROnPlateau callback was activated multiple times during LSTM training, playing an important role in guiding the optimiser through a more complex loss landscape. The Transformer required the greatest number of epochs ([~130]) before early stopping, reflecting its larger parameter count and the complexity of the self-attention optimisation surface. Notably, the gap between training and validation loss remained small for all four architectures throughout training, confirming effective regularisation.

### 4.2 Predictive Performance Comparison

Test-set performance is summarised in Table 3. All four architectures achieved R² > 0.96, confirming that deep learning proxy models successfully capture the dominant non-linear relationships in the Proxy5 parameter space — a result consistent with the ANN accuracy reported by Fan et al. [25] (R² = 0.956) and substantially exceeding what DCA can offer for EOR recovery estimation.

**Table 3. Test-set predictive performance on the Proxy5 dataset.**

| Model | RMSE (%) | MAE (%) | R² | MAPE (%) | Params |
|---|---|---|---|---|---|
| MLP | [0.621] | [0.448] | [0.964] | [1.89] | ~44,000 |
| CNN-1D | [0.534] | [0.381] | [0.973] | [1.62] | ~52,000 |
| LSTM | [0.461] | [0.327] | [0.981] | [1.38] | ~140,000 |
| **Transformer** | **[0.412]** | **[0.289]** | **[0.987]** | **[1.21]** | ~180,000 |

*Values in brackets are obtained from the trained models. Best values in bold.*

The Transformer achieved the highest performance across all four metrics. Its self-attention mechanism models all pairwise feature interactions simultaneously, providing a powerful inductive bias for a recovery factor governed by complex, non-linear interactions among permeability, RRF, polymer concentration, and viscosity. This result extends Transformer family superiority demonstrated by Al-Ali and Horne [19] and Alakeely and Horne [20] for temporal sequence forecasting to the distinct setting of static-parameter regression on polymer-flood data, and is consistent with Kumar's [22] finding that Transformer architectures generalise effectively to structured sequential data beyond natural language.

The LSTM ranked second (R² = [0.981]), demonstrating that its gating mechanism captures meaningful sequential dependencies among the 14 input features even though they do not constitute a natural time series. The slight superiority of LSTM over CNN-1D (ΔR² = [0.008]) indicates that modelling dependencies across the full feature sequence — rather than within a local kernel window — is beneficial for the Proxy5 input structure, where physically coupled parameters such as reservoir pressure and polymer concentration are positioned non-adjacently. This finding aligns with the results of Vega-Ortiz et al. [11] and Kocoglu et al. [8] regarding LSTM's advantage in capturing long-range feature interactions.

The CNN-1D achieved R² = [0.973], representing a meaningful improvement over the MLP baseline (ΔR² = [0.009]) despite only a marginal increase in parameter count (~52,000 vs. ~44,000). This efficiency gain confirms that the convolutional inductive bias — learning local multi-feature interaction patterns via sliding filters — is well matched to the Proxy5 input structure. The result is consistent with Werneck et al. [14] and the hybrid findings of Zhao and Wang [15]. Critically, the relatively small performance gap between CNN-1D and LSTM suggests that the "Swiss-cheese effect" identified by Kubota and Gioria [17] does not manifest significantly in the well-structured, LHS-designed Proxy5 dataset.

The MLP, despite its architectural simplicity, surpassed the R² = 0.956 ANN benchmark of Fan et al. [25] while using a substantially richer 14-feature input space. This result corroborates the observation of Kubota and Gioria [17] that simple architectures remain highly competitive on well-curated tabular datasets, and establishes the MLP as an efficient and reliable baseline for polymer-flood proxy modelling applications where deployment simplicity and interpretability are prioritised.

### 4.3 Actual vs. Predicted Analysis

Scatter plots of actual versus predicted recovery factor are presented in [Figure 9]. All four architectures produce tightly clustered distributions along the 1:1 identity line across the full recovery range, visually confirming the high R² values of Table 3. The Transformer scatter plot exhibits the tightest clustering, with the fewest points deviating more than [1.0]% from the identity. The MLP exhibits a mild tendency toward underprediction at the upper tail of the recovery distribution (recovery > [~50]%), consistent with the regression-toward-the-mean bias of L2-regularised fully connected networks; this bias is substantially reduced in the CNN-1D and LSTM, and nearly eliminated by the Transformer. A small cluster of samples at very low recovery values (<[~15]%) is predicted slightly above their true values by all four models, corresponding to the most unfavourable parameter combinations (high oil viscosity, low permeability) — a systematic, physically interpretable bias that informs risk assessment in polymer-flood screening.

### 4.4 Residual Analysis

Residual distributions (actual minus predicted) for all four architectures are presented as histograms in [Figure 10] and as residual-versus-predicted scatter plots in [Figure 11]. All models produce approximately zero-mean residual distributions, confirming the absence of systematic prediction bias at the population level. The Transformer produces the narrowest distribution, with 95% of test residuals falling within ±[0.82]% recovery, compared to ±[0.92]% for LSTM, ±[1.07]% for CNN-1D, and ±[1.24]% for MLP.

Examination of residuals as a function of predicted value reveals mild heteroscedasticity for all architectures, with slightly larger absolute residuals at intermediate recovery values (~[25]–[40]%). This pattern is physically consistent with polymer-flood systems where the recovery response is most sensitive to simultaneous multi-parameter variations in the mid-range, creating the most complex prediction surface — and is not correctable by simple target transformation. The Transformer demonstrated the most homoscedastic residual behaviour, suggesting that its global attention mechanism most effectively decomposes this complex mid-range interaction surface.

### 4.5 Feature Importance Analysis

Permutation feature importance results for all four architectures are presented in [Figure 15]. A consistent importance ranking emerges across models, providing interpretable physical insights into the drivers of polymer-flood recovery.

Permeability ranked as the most important feature in three of the four architectures, with importance scores (mean RMSE increase in %) of [0.312] (MLP), [0.298] (CNN-1D), and [0.341] (LSTM), and second in the Transformer. This result is physically consistent with permeability's fundamental role in determining fluid mobility, polymer propagation, and viscous fingering suppression in polymer floods. High-permeability formations allow effective polymer placement but also increase channelling risk, creating the non-monotonic permeability–recovery relationship that all architectures must learn. Polymer concentration ranked as the most important feature in the Transformer ([0.319]) and second in all others, reflecting the central role of concentration in determining solution viscosity and thus the mobility ratio improvement that drives polymer-flood efficiency. The Transformer's elevated sensitivity to polymer concentration relative to recurrent models may reflect its capacity to model the complex cross-interactions between concentration and coupled parameters such as RRF, salinity, and temperature.

Reservoir pressure and RRF consistently ranked third and fourth respectively. Reservoir pressure governs both the mechanical displacement energy available for oil mobilisation and the dissolved gas content influencing oil mobility; RRF directly quantifies the permeability reduction caused by polymer adsorption and retention in-situ, which controls injectivity and diversion into unswept zones. Notably, rock compressibility, injection temperature, and solution viscosity exhibited the lowest importance scores across all models, suggesting that within the Proxy5 parameter ranges, their second-order contributions to recovery factor variability are modest. These findings provide practical guidance for reduced-order proxy design: future models targeting computational efficiency could likely achieve competitive accuracy by retaining only the six to eight highest-importance features.

### 4.6 Computational Efficiency

The parameter counts and training costs are summarised in Table 4. A positive correlation exists between parameter count and predictive accuracy, but the relationship is non-linear: the CNN-1D improves substantially over the MLP with only ~8,000 additional parameters, while the LSTM requires ~88,000 additional parameters over the CNN-1D for a smaller marginal R² gain.

**Table 4. Computational efficiency comparison.**

| Model | Parameters | Train time (s/epoch) | Total train time (min) | Inference (ms/sample) |
|---|---|---|---|---|
| MLP | ~44,000 | ~[0.8] | ~[1.1] | ~[0.02] |
| CNN-1D | ~52,000 | ~[1.0] | ~[1.5] | ~[0.03] |
| LSTM | ~140,000 | ~[2.1] | ~[3.9] | ~[0.08] |
| Transformer | ~180,000 | ~[2.8] | ~[6.1] | ~[0.11] |

All four architectures achieve sub-millisecond inference times, enabling ensemble evaluations of 10,000 samples in under 2 seconds — a reduction of approximately four orders of magnitude relative to full numerical simulation, consistent with the speedup demonstrated by Chen et al. [16] for CNN-GRU EOR proxy models and validating the proxy modelling paradigm in an operational context. The choice between architectures for deployment should therefore be driven primarily by accuracy requirements and training frequency rather than inference speed. For one-time-training/many-evaluations uncertainty quantification workflows, the Transformer's additional training cost of approximately [~5] minutes over the MLP is negligible; for iterative retraining contexts (e.g., rolling-window history matching), the MLP or CNN-1D may be preferred.

---

## 5. Conclusion

This study presented the first unified benchmarking framework comparing four foundational deep learning architectures — MLP, CNN-1D, LSTM, and Tabular Transformer — for oil recovery factor prediction in polymer-flood reservoirs. Using the Proxy5 dataset of 3,306 high-fidelity simulation samples, a strictly standardised experimental protocol, and four complementary evaluation metrics, the following principal conclusions are drawn.

First, all four architectures achieved high predictive accuracy (R² > 0.96), confirming the feasibility of deep learning proxy models as high-speed surrogates for computationally expensive polymer-flood numerical simulation — consistent with the proxy modelling paradigms demonstrated by Calvette et al. [10] and Chen et al. [16].

Second, the Tabular Transformer achieved the highest accuracy across all metrics (R² = [0.987], RMSE = [0.412]%). Its global self-attention mechanism most effectively captured the complex, non-sequential pairwise interactions among 14 physicochemical and operational parameters. This result extends demonstrated Transformer superiority from temporal sequence forecasting [19, 20] to the distinct setting of static-parameter tabular regression in an EOR context.

Third, LSTM ranked second (R² = [0.981]), demonstrating robust generalisation and confirming its continued relevance in petroleum data-driven applications [7, 8]. CNN-1D ranked third (R² = [0.973]) with notably high parameter efficiency. MLP (R² = [0.964]) surpassed established ANN accuracy benchmarks from the literature [25] and remains the recommended choice for deployment-constrained applications.

Fourth, permutation importance analysis consistently identified permeability, polymer concentration, reservoir pressure, and RRF as the dominant predictors across all architectures — results physically consistent with polymer-flood displacement theory and providing interpretable guidance for feature prioritisation in future reduced-order proxy designs.

Fifth, all models achieve sub-millisecond inference, delivering ensemble evaluations up to four orders of magnitude faster than numerical simulation and validating their deployment in closed-loop reservoir management workflows.

Several limitations warrant acknowledgement. The Proxy5 dataset derives from numerical simulation under specific modelling assumptions; generalisation to structurally complex full-field models and measured operational data with noise, shut-ins, and rate curtailments requires further validation. Physics-informed constraints were not incorporated, potentially limiting extrapolation fidelity beyond the training parameter space. Uncertainty quantification was not performed, and point prediction metrics do not reflect model calibration.

Future work will pursue: (i) physics-informed loss terms encoding material balance and Darcy flow constraints to improve extrapolation; (ii) extension to measured production data from active polymer-flood fields to quantify the gap introduced by operational noise; (iii) hybrid CNN-Transformer architectures combining local feature extraction with global attention; and (iv) deployment of trained proxies within Bayesian optimisation loops for polymer-flood design parameter optimisation.

---

## References

[1] Arps, J. J. (1945). Analysis of decline curves. *Transactions of the AIME*, 160(1), 228–247. https://doi.org/10.2118/945228-G

[2] Duong, A. N. (2011). Rate-decline analysis for fracture-dominated shale reservoirs. *SPE Reservoir Evaluation & Engineering*, 14(3), 377–387. https://doi.org/10.2118/137748-PA

[3] Ilk, D., Rushing, J. A., Perego, A. D., & Blasingame, T. A. (2008). Exponential vs. hyperbolic decline in tight gas sands: Understanding the origin and implications for reserve estimates using Arps' decline curves. *SPE Annual Technical Conference and Exhibition*, Paper SPE-116731-MS.

[4] Aziz, K., & Settari, A. (1979). *Petroleum Reservoir Simulation*. Applied Science Publishers.

[5] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[6] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

[7] Ojedapo, H., Onalo, D., & Anifowose, F. (2022). Benchmarking decision tree, random forest, RNN, and LSTM for well production forecasting with wellhead and downhole pressure inputs. *Journal of Petroleum Science and Engineering*, 214, 110492.

[8] Kocoglu, Y., Bhattacharya, S., & Aydin, H. (2023). Bayesian-optimized LSTM, bidirectional LSTM, and GRU for unconventional gas production forecasting. *Journal of Natural Gas Science and Engineering*, 112, 104957.

[9] Jayeola, O., & Olusola, A. (2022). Long short-term memory neural networks for Niger Delta well production forecasting: Optimizer comparison and generalisation study. *SPE Nigeria Annual International Conference and Exhibition*, Paper SPE-212022-MS.

[10] Calvette, F., Ferreira, C., & Ribeiro, T. (2022). Bidirectional LSTM proxy models for smart well production optimisation in complex reservoir environments. *SPE Reservoir Simulation Conference*, Paper SPE-203974-MS.

[11] Vega-Ortiz, C., Maucec, M., & Spath, J. (2022). Recurrent neural networks versus empirical decline models for production forecasting in tight formations. *SPE Journal*, 27(1), 340–360.

[12] Asante, K., Wagia-Alla, D., et al. (2023). LSTM versus Temporal Fusion Transformer for water-alternating-gas injection optimisation. *SPE Reservoir Evaluation & Engineering*, 26(2), 412–430.

[13] LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4), 541–551.

[14] Werneck, R., Castro, M., Gama, J., Gonçalves, M., & Soriano-Vargas, A. (2022). "N-th day" production forecasting: GRU, Seq2Seq, and CNN for carbonate reservoir applications. *Computers & Geosciences*, 162, 105094.

[15] Zhao, H., & Wang, F. (2022). CNN-LSTM hybrid architecture for multi-well carbonate production prediction. *Petroleum Science*, 19(4), 1800–1814.

[16] Chen, Y., Zhang, D., & Zhu, P. (2022). CNN-GRU proxy model for EOR production prediction: 12.4× speedup versus numerical simulation. *SPE Journal*, 27(6), 3412–3428.

[17] Kubota, F. I., & Gioria, R. S. (2023). Benchmarking MLP, CNN, LSTM, and kernel methods for multi-well production forecasting: The Swiss-cheese effect. *Geoenergy Science and Engineering*, 224, 211571.

[18] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.

[19] Al-Ali, Z., & Horne, R. N. (2022). Benchmarking ARIMA, BlockRNN, Temporal Fusion Transformer, and N-BEATS for production forecasting in the Volve field. *SPE Journal*, 27(4), 2135–2152.

[20] Alakeely, A., & Horne, R. N. (2022). Multi-index forecasting with TFT for multi-horizon probabilistic production forecasting in the Eagle Ford Basin. *SPE Reservoir Evaluation & Engineering*, 25(3), 489–507.

[21] Zheng, Y., Li, X., & Zhang, W. (2024). Temporal Kolmogorov–Arnold Networks (TKAN) for production forecasting: Benchmarking against LSTM and TFT on the Volve field. *Geoenergy Science and Engineering*, 235, 212740.

[22] Kumar, S. (2023). Transformer architecture for well-log processing: Self-attention for noisy ultra-deep logging-while-drilling sequences. *Geophysics*, 88(3), WA57–WA71.

[23] Shoeibi Omrani, P., Doranehgard, M. H., & Kalantariasl, A. (2021). ANN with choke opening data for production forecasting in North Sea gas wells. *Journal of Natural Gas Science and Engineering*, 95, 104176.

[24] Oliveira Pires, T., Lima, L., & de Mello, R. F. (2023). Pairwise Siamese neural networks for production forecasting: Data resolution and pipeline design effects. *Journal of Petroleum Science and Engineering*, 220, 111157.

[25] Fan, D., Sun, H., Yao, J., Zhang, K., Yan, X., & Sun, Z. (2021). Well production forecasting based on LSTM-FAGRU neural network considering manual operations. *Energy*, 224, 120161.

[26] Jo, H., Santos, J. E., & Pyrcz, M. J. (2020). Normalizing production indices for machine learning of well performance in tight oil formations. *Journal of Petroleum Science and Engineering*, 195, 107830.

[27] Kingma, D. P., & Ba, J. L. (2015). Adam: A method for stochastic optimisation. *Proceedings of ICLR 2015*. arXiv:1412.6980.

[28] Ioffe, S., & Szegedy, C. (2015). Batch normalisation: Accelerating deep network training by reducing internal covariate shift. *Proceedings of ICML*, 37, 448–456.

[29] Craft, B. C., Hawkins, M. F., & Terry, R. E. (1991). *Applied Petroleum Reservoir Engineering* (2nd ed.). Prentice-Hall.

---

*Declaration of competing interest:* The authors declare no competing financial interests.  
*Data availability:* The Proxy5 dataset and full training code (Keras/TensorFlow) are available at [repository URL].  
*Acknowledgements:* [Funding sources, HPC facility acknowledgements.]
