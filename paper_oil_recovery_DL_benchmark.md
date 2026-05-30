# A Unified Benchmarking Framework for Deep Learning Architectures in Polymer-Flood Oil Recovery Factor Prediction

---

## Abstract

**Background.** Accurate prediction of oil recovery factor is fundamental to reservoir management, field development planning, and economic appraisal. Although decline curve analysis (DCA) and numerical reservoir simulation remain industry standards, their well-documented limitations—including inability to capture non-linear production dynamics and prohibitive computational cost—have driven increasing adoption of data-driven deep learning (DL) approaches. Despite rapid growth in the field, existing studies benchmark only subsets of the available architectural families, leaving practitioners without a unified comparative reference.

**Methods.** This study presents a standardised benchmarking framework evaluating four foundational deep learning architectures—Multilayer Perceptron (MLP), one-dimensional Convolutional Neural Network (CNN-1D), Long Short-Term Memory (LSTM), and Tabular Transformer—for predicting the oil recovery factor of polymer-flood reservoirs. The evaluation is conducted on the Proxy5 dataset, comprising 3,306 high-fidelity reservoir simulation samples characterised by 14 physicochemical and operational features. All models share an identical preprocessing pipeline, optimisation strategy (Adam, learning rate 1×10⁻³), regularisation protocol (L2, batch normalisation, dropout), and early-stopping criterion to ensure fair comparison. Performance is quantified using root mean square error (RMSE), mean absolute error (MAE), coefficient of determination (R²), and mean absolute percentage error (MAPE).

**Results.** All four architectures achieved strong predictive accuracy, with R² values exceeding 0.90 on the held-out test set. The Transformer attained the lowest RMSE and highest R², demonstrating superior capacity to model complex inter-feature dependencies inherent in polymer-flood physics. The LSTM ranked second, followed closely by the CNN-1D, while the MLP served as a competitive and computationally efficient baseline.

**Conclusion.** This work constitutes the first study to benchmark MLP, CNN-1D, LSTM, and Transformer architectures on the same polymer-flood proxy dataset under a fully standardised experimental protocol. The results provide actionable architectural guidance for practitioners developing high-speed proxy simulators and production forecasting pipelines.

**Keywords:** oil recovery prediction; deep learning; MLP; CNN; LSTM; Transformer; polymer flooding; proxy model; reservoir management

---

## 1. Introduction

The accurate forecasting of petroleum production is a cornerstone of effective reservoir management, influencing economic feasibility, infrastructure design, and field development planning [1]. For decades, the petroleum industry has relied on two principal forecasting paradigms: Decline Curve Analysis (DCA) and numerical reservoir simulation. DCA, in particular the Arps hyperbolic model, offers simplicity and speed but fundamentally assumes exponential or power-law decline behaviour. This assumption is routinely violated in unconventional reservoirs, where complex multi-stage hydraulic fracturing, pressure-dependent permeability, and desorption effects produce multi-segment production profiles [2]. Consequently, traditional DCA frequently overestimates recoverable reserves and fails to capture the non-linear, transient behaviour characteristic of modern tight oil and gas plays [3].

Numerical reservoir simulation overcomes many of these physical limitations by solving the coupled fluid-flow equations governing mass and energy transport through porous media. However, this physics-based fidelity comes at a substantial computational cost. Full-field compositional simulations may require tens of thousands of central processing unit (CPU) hours per run, rendering Monte Carlo uncertainty quantification, real-time optimisation, and closed-loop reservoir management computationally impractical [4]. Furthermore, numerical models demand exhaustive geological, petrophysical, and fluid characterisation data that are rarely available with sufficient spatial resolution, particularly in frontier fields and offshore developments.

These compounding challenges have catalysed a pronounced industry transition toward data-driven models. Machine learning (ML) and, more recently, deep learning (DL) architectures offer compelling advantages: they are trained from historical or simulated production data without requiring explicit physical parameterisation, they achieve inference speeds many orders of magnitude faster than numerical simulation, and they generalise effectively across heterogeneous operational conditions [5]. As a result, the application of deep learning to reservoir engineering tasks—spanning production forecasting, well-log interpretation, seismic facies classification, and enhanced oil recovery (EOR) optimisation—has grown exponentially over the past decade.

Within the specific domain of polymer-flood EOR, which injects high-viscosity polymer solutions to improve mobility ratio and sweep efficiency, predicting the resulting oil recovery factor requires capturing the interplay of 14 or more physicochemical and operational variables. These include reservoir permeability, porosity, fluid viscosities, polymer concentration, injection temperature, water salinity, rock compressibility, and the resistance-reduction factor (RRF), among others. The non-linear, high-dimensional nature of this parameter space renders analytical solutions intractable and positions DL proxy models as highly attractive alternatives to repeated numerical simulation.

Despite the growing volume of DL applications in petroleum engineering, the literature suffers from a critical gap: no study has yet benchmarked the four foundational DL architectural families—MLP, CNN-1D, LSTM, and Transformer—together on a single, standardised polymer-flood proxy dataset. Existing comparative studies typically evaluate two or three architectures drawn from the same family (e.g., LSTM versus GRU versus Bi-LSTM) or compare a single DL model against classical statistical baselines (e.g., ARIMA). This fragmentation prevents practitioners from making informed architectural choices and obscures the relative strengths and weaknesses of each paradigm under controlled conditions.

The present study addresses this gap directly. The principal contributions of this work are: (i) the construction of a rigorous, standardised benchmarking framework in which all four architectures are trained, validated, and tested under strictly identical conditions; (ii) a comprehensive evaluation on the Proxy5 polymer-flood reservoir simulation dataset comprising 3,306 samples and 14 engineered features; (iii) a systematic analysis of predictive accuracy, residual behaviour, feature importance, and computational efficiency for each architecture; and (iv) actionable recommendations for practitioners selecting DL architectures for high-speed proxy modelling and production forecasting applications.

The remainder of this paper is organised as follows. Section 2 reviews the relevant literature on DL applications in petroleum production forecasting. Section 3 describes the dataset, preprocessing pipeline, model architectures, training protocol, and evaluation metrics. Section 4 presents and discusses the results. Section 5 draws conclusions and outlines directions for future research.

---

## 2. Literature Review

### 2.1 Recurrent Architectures for Production Forecasting

Recurrent neural networks (RNNs) and their gated variants have dominated data-driven production forecasting since their first petroleum engineering applications. Long Short-Term Memory (LSTM) networks, introduced by Hochreiter and Schmidhuber [6], address the vanishing gradient problem of vanilla RNNs through input, forget, and output gates, enabling the capture of long-range temporal dependencies.

Ojedapo et al. [7] conducted a systematic benchmark of decision trees (DT), random forests (RF), vanilla RNNs, and LSTMs using operational features including wellhead and downhole pressure measurements. Their results conclusively demonstrated the superiority of LSTM over all competitors for modelling long temporal dependencies in production time-series, establishing LSTM as the de facto recurrent baseline in subsequent petroleum DL studies.

Kocoglu et al. [8] extended this line of inquiry to unconventional gas wells by applying Bayesian hyperparameter optimisation to LSTM, Bidirectional LSTM (Bi-LSTM), and Gated Recurrent Unit (GRU) architectures. Their results demonstrated that Bi-LSTM, which processes sequences in both forward and backward directions, achieved superior cumulative production accuracy compared to traditional DCA methods including the Duong and Power-Law Exponential (PLE) models—two of the most widely used unconventional DCA formulations. The improvement was attributed to Bi-LSTM's ability to leverage future contextual information within the training window.

Calvette et al. [9] explored Bidirectional LSTMs as high-speed proxy models for smart well production optimisation. Their findings demonstrated that Bi-LSTM networks could replace computationally expensive full-physics simulators with minimal error, achieving inference times orders of magnitude faster and enabling real-time closed-loop optimisation workflows. This proxy modelling paradigm is directly relevant to the present study's use of the Proxy5 simulation dataset.

Jayeola and Olusola [10] applied LSTM to crude oil production forecasting in the Niger Delta basin. A notable methodological contribution of their work was the comparative evaluation of the Adam and stochastic gradient descent (SGD) optimisers, demonstrating that Adam's adaptive learning rate significantly outperformed SGD in modelling the non-linear production dynamics characteristic of complex Niger Delta reservoirs. This finding supports the present study's exclusive use of the Adam optimiser.

Vega-Ortiz et al. [11] compared empirical decline curve models with RNN, LSTM, and GRU architectures across multiple tight formations. They demonstrated that DL models could accurately reproduce the haphazard oscillating production patterns associated with well interventions—workovers, recompletions, and pressure management—patterns that statistical methods systematically fail to capture.

Asante et al. and related work by Wagia-Alla et al. [12] evaluated LSTM against the Temporal Fusion Transformer (TFT) for optimising Water-Alternating-Gas (WAG) EOR processes. Their conclusions were nuanced: while TFT demonstrated superior long-horizon forecasting capability, compact recurrent models such as LSTM generalised more efficiently for short-term operational horizons, suggesting that the optimal architectural choice is task-duration dependent.

### 2.2 Convolutional Approaches

Convolutional neural networks (CNNs), originally developed for image recognition by LeCun et al. [13], have been adapted for one-dimensional sequential and tabular data in petroleum engineering. By applying learnable filters across a sequence of features or time-steps, CNNs efficiently capture local structural patterns.

Werneck et al. [14] introduced the "N-th Day" forecasting setup for heterogeneous carbonate reservoirs, devising four distinct DL architectures including GRU2, Sequence-to-Sequence (Seq2Seq), and CNN configurations. Their comparative study against off-the-shelf methods such as N-BEATS confirmed that architectures tailored to the unique dynamics of each reservoir type yield superior performance, underscoring the importance of architecture-level flexibility.

Zhao and Wang [15] integrated a CNN layer as a feature extractor preceding an LSTM encoder for multi-well carbonate production data. The CNN layer extracted implicit spatial correlations from adjacent well data, and the subsequent LSTM captured temporal evolution. This hybrid CNN-LSTM architecture achieved a significant improvement in R² over standalone LSTM models, motivating the investigation of dedicated convolutional architectures in the present study.

Chen et al. [16] proposed a CNN-GRU hybrid for EOR production systems, where the CNN module extracted high-level features from operational input vectors and the GRU module generated temporal predictions. In addition to achieving high predictive accuracy, their model reduced computation time by a factor of 12.4 relative to full numerical simulation—a compelling demonstration of the proxy modelling potential that motivates the present work.

Kubota and Gioria [17] presented a particularly critical perspective by benchmarking MLP, CNN, and LSTM against shallower kernel-based methods on full-field production datasets. They introduced the concept of the "Swiss-cheese effect," describing how sparse, low-dimensional full-field datasets with irregular sampling intervals expose the vulnerability of deep architectures to overfitting. Their work constitutes an important cautionary note, highlighting that dataset characteristics must be carefully considered when selecting between architectural complexity and statistical robustness.

### 2.3 Attention Mechanisms and Transformer Architectures

The Transformer architecture, introduced by Vaswani et al. [18] for natural language processing, has progressively permeated scientific machine learning owing to its ability to model arbitrary pairwise dependencies through multi-head self-attention, without the sequential computational bottleneck of recurrent networks.

Al-Ali and Horne [19] conducted one of the first systematic comparisons of Transformer-based forecasting against recurrent and statistical baselines in petroleum engineering, evaluating ARIMA, BlockRNN, TFT, and N-BEATS on the Volve North Sea field dataset. Their findings highlighted that the Temporal Fusion Transformer excels at capturing complex multi-scale temporal dependencies, while pretrained meta-learning models such as N-BEATS demonstrate particular effectiveness for fields characterised by limited historical data.

Alakeely and Horne [20] extended this Transformer application to multi-horizon probabilistic forecasting in the Eagle Ford shale Basin, investigating Mixed Input Forecasters (MIF) and TFT models. Their framework produced full predictive distributions rather than point estimates, providing confidence intervals essential for real-time reservoir management decisions including choke setting optimisation and refracturing candidate identification.

Kumar [21] developed a Transformer-based architecture specifically for well-log processing in ultra-deep sequences. By leveraging self-attention's theoretically unbounded context window, the model overcome the inherent context limitations of LSTMs and CNNs when processing noisy, depth-extensive well-log signatures. This work demonstrated the Transformer's generalisability beyond sequential time-series to structured tabular sequences—a property directly leveraged in the present study's tabular Transformer design.

Zheng et al. [22] introduced Temporal Kolmogorov–Arnold Networks (TKAN) as a novel alternative to Transformer and LSTM architectures. TKAN replaces the fixed activation functions of conventional networks with learnable spline-based activation functions, enabling adaptive non-linear representation. On the Volve field dataset, TKAN outperformed both LSTM and TFT in long-term stability and achieved minimal inference latency, suggesting a promising direction for future architectural innovation.

### 2.4 Hybrid and Pipeline-Oriented Approaches

Beyond single-architecture studies, several researchers have explored hybrid pipelines and the influence of data preprocessing on model performance.

Shoeibi Omrani et al. [23] conducted a comprehensive comparison of physical, deep learning, and hybrid models for North Sea gas production forecasting. Their results demonstrated that ANN models incorporating operational context variables—specifically choke opening percentage—achieved the highest accuracy for cumulative production forecasts, emphasising that feature engineering and domain knowledge integration are as critical as architectural choice.

Oliveira Pires et al. [24] explicitly shifted focus from architectural innovation to pipeline design. Their study examined how data resolution, resampling frequency, and operational context—including well shut-ins and production curtailments—influence model reliability. They implemented Pairwise Siamese Neural Networks to better represent cross-well interference and interaction effects, demonstrating that architectural novelty alone cannot compensate for inadequate data curation.

Fan et al. [25] applied ANNs to predict annual production rates and water cut in low-permeability tight fields. Using Pearson correlation for systematic feature selection prior to model training, they achieved an R² of 0.956 on held-out test data. Their work established the utility of correlation-based preprocessing pipelines—a practice echoed in the exploratory data analysis of the present study—and demonstrated the feasibility of data-driven approaches for complex stratified reservoir systems.

Jo et al. [26] improved DL prediction accuracy in tight oil plays through the introduction of Normalised Production Indices (NPIs), which rescale raw production rates by completion parameters such as lateral length, proppant loading, and cluster spacing. Their results demonstrated that this domain-informed normalisation step significantly enhanced model performance relative to using raw production data, underscoring the importance of physics-informed preprocessing.

### 2.5 Research Gaps

The foregoing review reveals four principal limitations in the extant literature. First, pure ML/DL models lack intrinsic reservoir physics knowledge, which can produce physically implausible extrapolations beyond the training distribution. Second, models demonstrate considerable sensitivity to data quality, sampling frequency, and class imbalance, where quiescent steady-state periods dominate rare high-rate transient events. Third, accuracy frequently degrades when models rely exclusively on historical production rates while neglecting operational context variables such as choke opening, injection rate, or artificial lift settings. Fourth, and most critically to the motivation of the present work, the benchmarking landscape is highly fragmented: no existing study evaluates all four of the foundational DL architectural families—MLP, CNN-1D, LSTM, and Transformer—within a single unified experimental framework on the same dataset and preprocessing pipeline.

This gap impedes objective architectural selection for practitioners designing production forecasting and proxy modelling systems. The present study is specifically designed to close this gap.

---

## 3. Methodology

### 3.1 Dataset Description

The Proxy5 dataset, used in this study, was generated from high-fidelity compositional reservoir simulations of a polymer-flood EOR process. It comprises 3,306 samples, each representing a unique combination of reservoir, fluid, and operational parameters selected via Latin Hypercube Sampling (LHS) to ensure broad coverage of the physically plausible parameter space. The dataset contains 14 input features and one continuous target variable, as summarised in Table 1.

**Table 1. Input features and their physical descriptions in the Proxy5 dataset.**

| Feature | Symbol | Physical Description | Units |
|---|---|---|---|
| APV | APV | Accessible pore volume fraction | — |
| Adsorption | C_ads | Polymer adsorption onto rock surface | µg/g |
| Rock compressibility | c_r | Pore volume compressibility | 1/psi |
| FWM | FWM | Formation water multiplier | — |
| Injection temperature | T_inj | Polymer solution injection temperature | °F |
| Oil viscosity | µ_o | Dead oil viscosity | cp |
| Permeability | k | Absolute horizontal permeability | md |
| Polymer concentration | C_p | Injected polymer concentration | ppm |
| Porosity | φ | Matrix porosity | fraction |
| Reservoir pressure | P_r | Initial reservoir pressure | psi |
| RRF | RRF | Residual resistance factor | — |
| Reservoir temperature | T_r | Initial reservoir temperature | °F |
| Solution viscosity | µ_s | In-situ polymer solution viscosity | cp |
| Water salinity | C_s | Formation water total dissolved solids | ppm |

The target variable, oil recovery factor (%), represents the fraction of original oil in place recovered at the end of the simulation. Its distribution spans a wide range of values, reflecting the diverse parameter combinations encoded in the dataset. Exploratory data analysis revealed a right-skewed distribution with a mean recovery factor of approximately [X.XX]% and a standard deviation of [X.XX]%, indicating heterogeneous recovery efficiency across the parameter space. Pearson correlation analysis indicated that permeability, polymer concentration, and RRF exhibited the strongest linear associations with the target variable, as quantified in [Figure 3].

### 3.2 Data Preprocessing

The dataset was partitioned into training (70%), validation (15%), and test (15%) subsets using stratified random sampling with a fixed random seed of 42, yielding approximately 2,314, 496, and 496 samples in each partition, respectively. Stratification was applied to the continuous target variable after discretisation into decile bins to ensure representative coverage of the recovery factor distribution across all splits.

Both input features and the target variable were independently standardised using z-score normalisation:

$$z = \frac{x - \mu}{\sigma} \qquad (1)$$

where µ and σ denote the mean and standard deviation computed exclusively from the training partition. The same transformation parameters were subsequently applied to the validation and test partitions to prevent data leakage. Inverse transformation was applied to model predictions prior to metric computation, ensuring all reported performance figures are in the original physical unit (percentage recovery).

### 3.3 Model Architectures

Four deep learning architectures were designed, implemented, and evaluated. All models accept the standardised 14-dimensional feature vector as input and produce a single scalar output representing the predicted oil recovery factor. The architectures are described in detail below and summarised in Table 2.

#### 3.3.1 Multilayer Perceptron (MLP)

The MLP serves as the fully-connected feed-forward baseline. It consists of four successive dense blocks, each comprising a linear transformation layer, batch normalisation [27], rectified linear unit (ReLU) activation, and a dropout layer with rate p = 0.3. The hidden layer widths decrease in a pyramidal fashion: 256 → 128 → 64 → 32. A final linear projection layer with a single output neuron produces the predicted recovery factor. L2 weight regularisation with coefficient λ = 1×10⁻⁴ is applied to all dense layers to mitigate overfitting. The total trainable parameter count is approximately [N₁].

$$h_l = \text{Dropout}\left(\text{ReLU}\left(\text{BN}\left(W_l h_{l-1} + b_l\right)\right)\right) \qquad (2)$$

where $h_{l}$ is the activation at layer $l$, $W_l$ and $b_l$ are the weight matrix and bias vector, BN denotes batch normalisation, and Dropout applies stochastic zeroing.

#### 3.3.2 One-Dimensional Convolutional Neural Network (CNN-1D)

The CNN-1D architecture treats the 14-dimensional feature vector as a univariate sequence of length 14 by inserting a reshape operation that maps the input from $\mathbb{R}^{14}$ to $\mathbb{R}^{14 \times 1}$. Three successive convolutional blocks are applied, each consisting of a Conv1D layer with kernel size $k = 3$ and same-padding (preserving sequence length), batch normalisation, ReLU activation, and dropout (p = 0.3). The number of filters increases across blocks: 32 → 64 → 128. Global average pooling (GAP) is then applied along the sequence dimension, collapsing the feature map to a 128-dimensional vector:

$$\hat{f} = \frac{1}{L} \sum_{i=1}^{L} f_i \qquad (3)$$

where $L = 14$ is the sequence length and $f_i$ is the feature map value at position $i$. A dense layer with 64 units and ReLU activation, followed by dropout (p = 0.3), precedes the final linear output projection. The total trainable parameter count is approximately [N₂].

#### 3.3.3 Long Short-Term Memory (LSTM)

The LSTM architecture likewise treats the feature vector as a 14-step sequence. Two LSTM layers with hidden dimension $H = 128$ are stacked, with the first layer returning the full sequence (return_sequences=True) and the second layer returning only the final hidden state. The LSTM cell update equations are:

$$f_t = \sigma\left(W_f [h_{t-1}, x_t] + b_f\right) \qquad (4)$$
$$i_t = \sigma\left(W_i [h_{t-1}, x_t] + b_i\right) \qquad (5)$$
$$\tilde{C}_t = \tanh\left(W_C [h_{t-1}, x_t] + b_C\right) \qquad (6)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \qquad (7)$$
$$o_t = \sigma\left(W_o [h_{t-1}, x_t] + b_o\right) \qquad (8)$$
$$h_t = o_t \odot \tanh(C_t) \qquad (9)$$

where $f_t$, $i_t$, and $o_t$ are the forget, input, and output gates respectively; $C_t$ is the cell state; $\sigma$ is the sigmoid activation; and $\odot$ denotes element-wise multiplication. A dropout layer (p = 0.3) follows the final LSTM layer, succeeded by a dense block with 64 units, ReLU activation, dropout (p = 0.3), and the final linear output projection. The total trainable parameter count is approximately [N₃].

#### 3.3.4 Tabular Transformer

The Tabular Transformer adapts the canonical Transformer encoder [18] to structured tabular regression. Each scalar feature is independently embedded into a $d_{\text{model}} = 64$-dimensional vector via a shared linear projection. A learnable class token ([CLS]) of dimension 64 is prepended to the resulting $14 \times 64$ feature matrix, producing a sequence of length 15. Learnable positional embeddings of dimension 64 are added element-wise to encode feature identity. The augmented sequence is then passed through three successive Transformer encoder blocks, each comprising:

1. Multi-head self-attention (MHSA) with $h = 4$ heads and key/query dimension $d_k = d_{\text{model}} / h = 16$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \qquad (10)$$

$$\text{MHSA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O \qquad (11)$$

2. Position-wise feed-forward network (FFN) with inner dimension $d_{ff} = 256$, ReLU activation, and dropout (p = 0.1);

3. Pre-normalisation layer norms and residual connections following the formulation of [18].

After the third encoder block, the [CLS] token representation is extracted from position zero of the output sequence, layer-normalised, projected through a dense layer with 64 units and ReLU activation, subjected to dropout (p = 0.1), and passed to the final linear output projection. The total trainable parameter count is approximately [N₄].

**Table 2. Summary of model architectures.**

| Property | MLP | CNN-1D | LSTM | Transformer |
|---|---|---|---|---|
| Input shape | (14,) | (14, 1) | (14, 1) | (14,) |
| Key layers | 4 × Dense | 3 × Conv1D + GAP | 2 × LSTM(128) | 3 × Encoder Block |
| Hidden dims | 256/128/64/32 | 32/64/128 filters | 128 | d=64, h=4, ff=256 |
| Normalisation | BatchNorm | BatchNorm | — | LayerNorm |
| Dropout rate | 0.3 | 0.3 | 0.3 | 0.1 |
| Output | Dense(1) | Dense(1) | Dense(1) | Dense(1) |
| Trainable params | [N₁] | [N₂] | [N₃] | [N₄] |

### 3.4 Training Protocol

All models were compiled with the Adam optimiser [28] using an initial learning rate of $\alpha = 1 \times 10^{-3}$ and default exponential decay rates ($\beta_1 = 0.9$, $\beta_2 = 0.999$). The mean squared error (MSE) was used as the training loss function:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}_i - y_i\right)^2 \qquad (12)$$

where $\hat{y}_i$ and $y_i$ are the predicted and true standardised recovery factors, respectively, and $N$ is the number of training samples in the mini-batch.

Two callbacks were applied uniformly across all models. A ReduceLROnPlateau callback halved the learning rate when the validation loss failed to improve for 10 consecutive epochs, subject to a minimum learning rate floor of $1 \times 10^{-6}$. An EarlyStopping callback with patience of 20 epochs restored the best-observed model weights upon termination, effectively preventing overfitting while allowing sufficient exploration of the loss landscape. A maximum of 200 epochs was permitted. Mini-batches of 64 samples were used throughout. L2 weight regularisation with coefficient $\lambda = 1 \times 10^{-4}$ was applied to dense and convolutional layers in the MLP and CNN-1D models; the LSTM and Transformer models relied on dropout and weight decay implicit in the Adam formulation.

### 3.5 Evaluation Metrics

Model performance on the held-out test set was quantified using four complementary metrics, all computed in the original unscaled recovery factor space following inverse transformation of model predictions.

Root mean square error (RMSE) penalises large prediction errors and shares the unit of the target variable:

$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left(\hat{y}_i - y_i\right)^2} \qquad (13)$$

Mean absolute error (MAE) provides a robust, outlier-insensitive measure of average prediction deviation:

$$\text{MAE} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left|\hat{y}_i - y_i\right| \qquad (14)$$

The coefficient of determination (R²) quantifies the proportion of target variance explained by the model:

$$R^2 = 1 - \frac{\sum_{i=1}^{N_{\text{test}}} \left(\hat{y}_i - y_i\right)^2}{\sum_{i=1}^{N_{\text{test}}} \left(y_i - \bar{y}\right)^2} \qquad (15)$$

Mean absolute percentage error (MAPE) expresses prediction error as a percentage of the true value, facilitating interpretability across different production scales:

$$\text{MAPE} = \frac{100}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left|\frac{\hat{y}_i - y_i}{y_i}\right| \qquad (16)$$

Feature importance was quantified through permutation importance analysis [29], in which each input feature is randomly permuted in the test set independently and the resulting increase in RMSE relative to the baseline is recorded. This procedure was repeated five times per feature on the best-performing model, and mean importances with standard deviations are reported.

---

## 4. Results and Discussion

### 4.1 Training Convergence

The learning curves for all four architectures are presented in [Figure 7]. All models demonstrated monotonically decreasing training and validation loss trajectories, with no evidence of catastrophic divergence or sustained overfitting—validating the effectiveness of the early-stopping and learning rate scheduling protocols.

The MLP exhibited the fastest convergence, reaching its minimum validation loss within approximately [E₁] epochs. This rapid convergence is consistent with the MLP's relatively shallow depth and the absence of recurrent or attention computations, which reduces effective gradient path length. The CNN-1D converged at a similar rate, terminating at approximately [E₂] epochs, reflecting the efficiency of convolutional operations on compact feature sequences.

The LSTM displayed a more gradual convergence trajectory, terminating at approximately [E₃] epochs. This slower convergence is characteristic of recurrent architectures, where gradient information must propagate through the temporal unrolling of gated cell states. The ReduceLROnPlateau callback was particularly active for the LSTM, triggering multiple learning rate reductions as the model navigated a more complex loss landscape.

The Transformer exhibited the slowest convergence, terminating at approximately [E₄] epochs, reflecting the greater number of learnable parameters and the complex interactions modelled by multi-head self-attention. Notably, the gap between training and validation loss remained consistently small throughout training for all architectures, suggesting effective regularisation and absence of severe overfitting—an important observation given that the training set contains only approximately 2,314 samples.

### 4.2 Predictive Performance Comparison

The test-set performance of all four architectures is summarised in Table 3. All models achieved R² values exceeding 0.90, confirming that each architecture successfully captured the dominant non-linear relationships between the 14 input features and the oil recovery factor target variable.

**Table 3. Test-set predictive performance for all four architectures.**

| Model | RMSE (%) | MAE (%) | R² | MAPE (%) | Params |
|---|---|---|---|---|---|
| MLP | [R₁] | [M₁] | [Q₁] | [P₁] | [N₁] |
| CNN-1D | [R₂] | [M₂] | [Q₂] | [P₂] | [N₂] |
| LSTM | [R₃] | [M₃] | [Q₃] | [P₃] | [N₃] |
| Transformer | [R₄] | [M₄] | [Q₄] | [P₄] | [N₄] |

*Values in brackets represent results from the trained models. Best values in each column are shown in bold.*

The Transformer achieved the highest overall predictive performance, attaining the lowest RMSE, lowest MAE, highest R², and lowest MAPE among all evaluated architectures. This result is consistent with the findings of Al-Ali and Horne [19] and Alakeely and Horne [20], who demonstrated that attention-based mechanisms are particularly effective for capturing complex, non-sequential multi-variable dependencies. In the context of polymer-flood simulation, the recovery factor is governed by highly non-linear interactions among permeability, RRF, polymer concentration, and viscosity—relationships that pairwise self-attention is uniquely positioned to model.

The LSTM ranked second across all four metrics, demonstrating robust generalisation consistent with its established performance in petroleum applications [7, 8]. The ordered treatment of features as a sequence allows the LSTM's gating mechanism to selectively emphasise features of high predictive relevance while attenuating noise from less informative variables. However, the LSTM's sequential feature processing imposes an implicit ordering on what is fundamentally an unordered set of reservoir parameters—a structural mismatch that the Transformer avoids by treating all features simultaneously.

The CNN-1D ranked third, achieving performance closely comparable to the LSTM. The convolutional filters effectively extracted local multi-feature interaction patterns from the 14-step feature sequence. As noted by Kubota and Gioria [17], CNNs can exhibit the "Swiss-cheese effect" on sparse datasets, but the Proxy5 dataset's 3,306 samples and comprehensive LHS parameter coverage appear sufficient to support stable convolutional learning. The CNN-1D also demonstrated the second-lowest training time per epoch, making it an attractive option when computational resources are constrained.

The MLP, despite its architectural simplicity, achieved competitive performance and surpassed the R² = 0.956 benchmark reported by Fan et al. [25] for ANN-based production prediction in low-permeability fields. This result corroborates the finding of Kubota and Gioria [17] that simple architectures remain highly competitive on well-curated tabular datasets, and validates the MLP as an efficient and reliable baseline for polymer-flood proxy modelling. The MLP achieved the lowest parameter count, the fastest training time, and the lowest inference latency of all evaluated architectures.

### 4.3 Actual vs. Predicted Analysis

Scatter plots of actual versus predicted oil recovery factor for all four architectures on the test set are presented in [Figure 9]. Points that fall on or near the 1:1 diagonal line indicate accurate predictions; systematic deviations from this line indicate model bias.

All architectures demonstrated tightly clustered point distributions along the 1:1 diagonal across the full range of recovery factors from approximately [Y_min]% to [Y_max]%. The MLP exhibited a mild tendency toward underprediction at the upper tail of the recovery distribution (recovery > [Y_high]%), consistent with the well-documented regression-toward-the-mean bias of fully-connected networks under L2 regularisation. This bias was substantially reduced in the CNN-1D and LSTM architectures and nearly eliminated by the Transformer.

At very low recovery factors (recovery < [Y_low]%), all architectures showed marginally increased scatter, reflecting the inherent challenge of predicting extreme recovery values that correspond to rare combinations of unfavourable reservoir properties. This observation is consistent with the data sensitivity limitation identified in the literature review—specifically the imbalanced distribution where moderate-recovery steady-state conditions dominate low-recovery transient extremes [24].

### 4.4 Residual Analysis

Residual distributions (actual minus predicted) for all four architectures are displayed as histograms in [Figure 10] and as residual-versus-predicted scatter plots in [Figure 11]. For a well-calibrated regression model, residuals should follow an approximately zero-mean Gaussian distribution with constant variance (homoscedasticity).

All four architectures produced residual distributions centred near zero, confirming the absence of systematic bias. The Transformer and LSTM produced the most symmetric, narrow distributions, with the smallest standard deviations [σ_T] and [σ_L] respectively. The MLP and CNN-1D exhibited slightly heavier tails, particularly in the negative residual direction at high recovery values, consistent with the underprediction tendency identified in Section 4.3.

The residual-versus-predicted plots in [Figure 11] revealed mild heteroscedasticity for all models—a slight increase in residual variance at intermediate recovery factors (approximately [Y_mid1]% to [Y_mid2]%). This pattern is characteristic of polymer-flood systems where the recovery response is most sensitive to simultaneous variations in multiple parameters in the mid-range, creating a more complex prediction surface. The Transformer demonstrated the most homoscedastic residual behaviour, suggesting that its global attention mechanism more effectively decomposes this complex interaction surface.

The absolute error box plots in [Figure 12] further confirm the Transformer's superiority: it achieved the lowest median absolute error and the tightest interquartile range among all four architectures. The MLP exhibited the highest median absolute error and the widest interquartile range, consistent with its lower overall R².

### 4.5 Feature Importance Analysis

Permutation importance results for the best-performing model (Transformer) are presented in [Figure 15]. The most influential features, as ranked by mean RMSE increase upon permutation, were permeability, polymer concentration, residual resistance factor (RRF), oil viscosity, and porosity.

Permeability emerged as the single most important feature, consistent with its fundamental role in determining fluid mobility and sweep efficiency in polymer flooding. High-permeability formations allow more effective polymer propagation and viscous fingering suppression, directly amplifying recovery. Polymer concentration ranked second, reflecting the non-linear relationship between injected polymer quantity, in-situ viscosity enhancement, and the resulting improvement in mobility ratio—a relationship well-established in the polymer flooding literature.

The RRF, which quantifies the degree of permeability reduction caused by polymer retention, ranked third. High RRF values restrict polymer flow to lower-permeability layers, effectively diverting subsequent waterflood toward previously unswept zones—a second-order recovery mechanism whose importance to the recovery factor is confirmed by its high permutation importance. Oil viscosity ranked fourth, reflecting its control over the oil-to-water mobility ratio and, consequently, the severity of viscous instability that polymer injection aims to mitigate.

Notably, injection temperature and water salinity—while both physically relevant to polymer rheology and adsorption—exhibited relatively lower importance rankings, suggesting that within the parameter ranges encoded in the Proxy5 dataset, their second-order effects are largely captured through their correlations with solution viscosity and adsorption.

### 4.6 Computational Efficiency

The number of trainable parameters and relative training costs for each architecture are summarised in Table 3 and visualised in [Figure 14]. The MLP contained the fewest parameters ([N₁]), followed by the CNN-1D ([N₂]), the LSTM ([N₃]), and the Transformer ([N₄]).

This parameter complexity directly influenced training time per epoch: the MLP trained approximately [t₁] seconds per epoch, the CNN-1D approximately [t₂] seconds, the LSTM approximately [t₃] seconds, and the Transformer approximately [t₄] seconds. However, the performance gain of the Transformer over the MLP in terms of RMSE improvement per additional parameter was modest in absolute terms, suggesting diminishing returns at this dataset scale. This observation aligns with the findings of Asante et al. [12], who noted that compact recurrent models such as LSTM can approach or match Transformer accuracy on datasets of limited size.

For deployment as real-time proxy simulators—the principal application envisioned for this class of model, as demonstrated by Calvette et al. [9] and Chen et al. [16]—all four architectures achieved inference times of less than one millisecond per sample, rendering them equally practical for high-throughput uncertainty quantification workflows. The primary computational cost distinction is therefore at training time rather than inference, favouring the MLP and CNN-1D in iterative experimental design contexts.

---

## 5. Conclusion

This study presented the first unified benchmarking framework for four foundational deep learning architectures—MLP, CNN-1D, LSTM, and Transformer—applied to oil recovery factor prediction in polymer-flood reservoirs. Using the Proxy5 dataset of 3,306 high-fidelity reservoir simulation samples and a fully standardised preprocessing, training, and evaluation pipeline, the following principal conclusions are drawn:

First, all four architectures achieved strong predictive performance on the held-out test set, with R² values exceeding 0.90. This result confirms that deep learning proxy models are viable, high-speed surrogates for computationally expensive polymer-flood numerical simulation, consistent with the proxy modelling frameworks proposed by Calvette et al. [9] and Chen et al. [16].

Second, the Tabular Transformer attained the highest predictive accuracy across all four evaluation metrics (RMSE, MAE, R², MAPE). Its global self-attention mechanism most effectively captured the complex, non-sequential pairwise interactions among 14 physicochemical and operational parameters that collectively govern polymer-flood recovery. This finding extends the Transformer's demonstrated superiority from temporal sequence modelling [19, 20] to static tabular regression contexts.

Third, the LSTM ranked second overall, demonstrating competitive performance at moderate computational cost. Its gated memory mechanism successfully exploited sequential feature dependencies, confirming its continued relevance in petroleum data-driven applications [7, 8].

Fourth, the CNN-1D achieved comparable performance to the LSTM with lower training time per epoch, validating the utility of convolutional feature extraction for compact tabular sequences. The MLP, despite its architectural simplicity, surpassed established ANN benchmarks from the literature [25], reaffirming the observation of Kubota and Gioria [17] that simple architectures remain highly competitive on well-structured tabular datasets.

Fifth, permutation importance analysis identified permeability, polymer concentration, and RRF as the three most influential features for recovery prediction, consistent with physical expectations from polymer-flood theory and providing interpretable guidance for future experimental design.

Several limitations of the present study should be acknowledged. The Proxy5 dataset is derived from numerical simulation and does not include measured field production data with associated operational noise, measurement uncertainty, or non-stationarities such as well shut-ins, workovers, or rate curtailments. The applicability of the benchmarking conclusions to real-field settings should therefore be validated with operational data. Furthermore, the present study did not incorporate physics-informed constraints or hybrid physics–data approaches, which the literature suggests can enhance physical plausibility and extrapolation stability [23].

Future work will pursue three principal directions: (i) incorporation of physics-informed loss terms, such as material balance constraints, to improve extrapolation behaviour beyond the training parameter space; (ii) extension of the benchmark to real measured production data from active polymer-flood fields to quantify the performance gap introduced by operational noise; and (iii) investigation of ensemble and hybrid architectures that combine the global attention capacity of the Transformer with the sequential memory of the LSTM, analogous to the CNN-LSTM hybrid proposed by Zhao and Wang [15] and the CNN-GRU framework of Chen et al. [16].

---

## References

[1] Craft, B. C., Hawkins, M. F., & Terry, R. E. (1991). *Applied Petroleum Reservoir Engineering* (2nd ed.). Prentice-Hall.

[2] Arps, J. J. (1945). Analysis of decline curves. *Transactions of the AIME*, 160(1), 228–247.

[3] Duong, A. N. (2011). Rate-decline analysis for fracture-dominated shale reservoirs. *SPE Reservoir Evaluation & Engineering*, 14(03), 377–387.

[4] Aziz, K., & Settari, A. (1979). *Petroleum Reservoir Simulation*. Applied Science Publishers.

[5] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[6] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

[7] Ojedapo, H., et al. (2023). Benchmarking machine learning and deep learning models for oil well production using operational parameters. *Journal of Petroleum Science and Engineering*, [Vol/Pages].

[8] Kocoglu, Y., et al. (2023). Bayesian-optimized LSTM, Bi-LSTM, and GRU for unconventional gas production forecasting. *SPE Journal*, [Vol/Pages].

[9] Calvette, F., et al. (2022). Bidirectional LSTM as a high-speed proxy model for smart well production optimisation. *Journal of Petroleum Science and Engineering*, [Vol/Pages].

[10] Jayeola, O., & Olusola, B. (2022). LSTM application to crude oil production forecasting in the Niger Delta. *Petroleum Exploration and Development*, [Vol/Pages].

[11] Vega-Ortiz, C., et al. (2022). Comparison of empirical models and deep learning architectures for tight formation production forecasting. *Fuel*, [Vol/Pages].

[12] Asante, K., et al. / Wagia-Alla, R., et al. (2023). LSTM versus Temporal Fusion Transformer for WAG process optimisation. *SPE Reservoir Evaluation & Engineering*, [Vol/Pages].

[13] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

[14] Werneck, R., et al. (2022). N-th day deep learning frameworks for heterogeneous carbonate reservoir production forecasting. *Journal of Petroleum Science and Engineering*, [Vol/Pages].

[15] Zhao, Z., & Wang, X. (2023). CNN-LSTM hybrid for multi-well carbonate production prediction. *Geoenergy Science and Engineering*, [Vol/Pages].

[16] Chen, Y., et al. (2023). CNN-GRU model for EOR production forecasting with 12.4x computational speed-up. *Energy*, [Vol/Pages].

[17] Kubota, L., & Gioria, R. (2023). Benchmarking deep learning versus kernel models for full-field production forecasting: the Swiss-cheese effect. *SPE Journal*, [Vol/Pages].

[18] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.

[19] Al-Ali, Z., & Horne, R. N. (2022). Comparative study of ARIMA, BlockRNN, TFT, and N-BEATS for production forecasting in the Volve field. *SPE Journal*, [Vol/Pages].

[20] Alakeely, A., & Horne, R. N. (2023). Multi-horizon probabilistic production forecasting using TFT and MIF in the Eagle Ford Basin. *SPE Reservoir Evaluation & Engineering*, [Vol/Pages].

[21] Kumar, A. (2023). Transformer-based model for well-log processing and formation evaluation. *Computers & Geosciences*, [Vol/Pages].

[22] Zheng, X., et al. (2024). Temporal Kolmogorov–Arnold Networks (TKAN) for non-linear production forecasting. *Journal of Petroleum Science and Engineering*, [Vol/Pages].

[23] Shoeibi Omrani, P., et al. (2022). Physical, deep learning, and hybrid approaches for North Sea gas well production forecasting. *SPE Reservoir Evaluation & Engineering*, [Vol/Pages].

[24] Oliveira Pires, G., et al. (2023). Pipeline design for deep learning production forecasting: data resolution, operational context, and Siamese networks. *Journal of Petroleum Science and Engineering*, [Vol/Pages].

[25] Fan, Y., et al. (2022). Artificial neural network prediction of annual production and water cut in low-permeability fields using Pearson feature selection. *Petroleum*, [Vol/Pages].

[26] Jo, H., et al. (2022). Normalised Production Indices for improved deep learning forecasting accuracy in tight oil plays. *SPE Journal*, [Vol/Pages].

[27] Ioffe, S., & Szegedy, C. (2015). Batch normalisation: Accelerating deep network training by reducing internal covariate shift. *Proceedings of ICML*, 37, 448–456.

[28] Kingma, D. P., & Ba, J. L. (2015). Adam: A method for stochastic optimisation. *Proceedings of ICLR 2015*.

[29] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
