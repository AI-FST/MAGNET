# MAGNET

MAGNET stands for **M**ulti-scale **A**daptive **G**ating **N**etwork with direc**T**ional optimization. It is an end-to-end framework for one-step-ahead financial time-series forecasting on noisy, non-stationary, multi-scale data. This repository accompanies the paper *Multi-Scale Financial Time-Series Forecasting with Structured Fusion and Direction-Aware Optimization*.

## Overview

Financial series contain persistent trends, periodic fluctuations, and abrupt high-frequency movements whose relative importance changes across market regimes. MAGNET addresses three linked problems: scale entanglement in a single encoder, signal dilution under static multi-scale fusion, and the misalignment between symmetric numerical losses and direction-sensitive volatility-state objectives.

The model separates the input into trend and seasonal components before multi-scale encoding, connects the two scales with trend-guided modulation and context-adaptive reweighting, and trains the full pipeline with a hybrid objective that combines mean squared error (MSE) with a magnitude-weighted sign-alignment term.

## Method

### Trend-Seasonal Decomposition

The input sequence is decomposed with a moving-average kernel into a smooth trend component and a residual seasonal component before the two branches are encoded in parallel.

### Legendre Projection Trend Encoder (LPTE)

LPTE maintains a fixed-dimensional memory through a predefined Legendre projection recurrence, applies a frequency-enhanced layer that suppresses high-frequency noise, and maps the enhanced memory to the shared hidden space. The projection matrices are fixed and contain no trainable parameters.

### Multi-Period 2D Seasonal Encoder (MPSE)

MPSE discovers the dominant periods of the seasonal component with an FFT, folds the one-dimensional sequence into two-dimensional tensors, applies parallel two-dimensional Inception-style convolutions over intraperiod and interperiod axes, and adaptively aggregates the period-specific outputs by their spectral amplitudes.

### Trend-Guided Seasonal Modulation (TGSM)

TGSM generates an elementwise gate from trend features and multiplies it with seasonal features. Because the gate is computed from trend information alone, the macro state controls how strongly each local fluctuation counts without replacing the seasonal feature with trend content.

### Context-Adaptive Scale Reweighting (CASR)

CASR flattens the full trend and modulated-seasonal feature maps, predicts two softmax-normalized scale weights from the complete observation window, and returns their weighted sum as the fused representation.

### Direction-Aware Optimization

Training uses the hybrid objective `L = (1 - lambda) * MSE + lambda * MSAL`. The Magnitude-Weighted Sign-Alignment Loss (MSAL) scores sign agreement between the prediction and the standardized target, weighted by target magnitude. Its forward pass keeps the hard sign function, while backpropagation uses a straight-through estimator based on a smooth `tanh(tau * y_hat)` surrogate. Here "direction" means whether standardized log realized volatility is above or below its training mean, not the buy or sell direction of an asset return.

## Datasets

The primary financial evaluation uses four datasets that differ in asset class, sampling frequency, and volatility dynamics:

| Dataset | Sampling | Description |
|---|---|---|
| CSI 300 | Daily | China A-share market index covering 300 large, liquid stocks; includes the 2008 global financial crisis, 2015 A-share turbulence, and the 2020 COVID-19 shock |
| S&P 500 | Daily | US large-cap equity index based on daily closing prices from December 1927 to November 2020 |
| Nasdaq-100 | Daily | US index dominated by large technology companies, providing complementary volatility dynamics |
| Bitcoin | Minute | Continuously traded cryptocurrency with extreme volatility and limited conventional fundamental anchors |

For the financial task, the two input features are log realized volatility and log volatility, and the target is the next-period log realized volatility. Samples are split chronologically into training, validation, and test sets. Features are standardized with training-set statistics only, so no future information enters preprocessing.

The Exchange, Traffic, and Weather benchmarks are included as auxiliary non-financial tests of cross-domain robustness. They retain their native forecasting targets and are evaluated only with numerical error metrics.

## Metrics

- MSE, RMSE, and MAE measure numerical accuracy; lower is better.
- Directional accuracy (DA) measures sign agreement in standardized target space; higher is better. It distinguishes high- and low-volatility states relative to the training mean and is not a return-direction accuracy.

## Main Results

MAGNET is compared with 13 recent baselines from Transformer, MLP, state-space, adaptation, objective-based, foundation-model, and diffusion families. It ranks first in all 12 financial dataset-metric cells. Reported test results are:

| Dataset | MSE | RMSE | MAE |
|---|---:|---:|---:|
| Nasdaq-100 | 0.1590 | 0.3987 | 0.2770 |
| CSI 300 | 0.1764 | 0.4233 | 0.2813 |
| S&P 500 | 0.1414 | 0.3761 | 0.2508 |
| Bitcoin | 0.1635 | 0.4043 | 0.2591 |

Compared with the second-best model for each dataset and metric, MAGNET achieves average relative reductions of 16.08%, 6.82%, and 6.08% in MSE, RMSE, and MAE, respectively. The largest relative gains appear on Bitcoin and on the CSI 300 and S&P 500 equity indices, while the Nasdaq-100 margins are smaller but still positive on every metric.

On the auxiliary non-financial benchmarks, MAGNET achieves the lowest MSE and RMSE on Exchange, Traffic, and Weather. Its MAE ranks second on Traffic and remains within 0.0029 of the best result on Exchange and Weather.

## Ablation and Mechanism Analysis

- Removing LPTE-MPSE replaces the dual-branch encoder with a single-scale model. This causes the largest degradation on the financial datasets; for example, Bitcoin MSE rises from 0.1635 to 0.3640.
- Removing TGSM-CASR replaces structured fusion with direct feature concatenation. Errors increase on all financial datasets, with the largest relative MSE increases on Bitcoin and the S&P 500.
- Removing MSAL trains with MSE alone. Numerical errors increase on all four financial datasets, and the relative contribution of each component is domain dependent; removing MSAL causes the largest degradation on Weather.
- Wavelet energy-fidelity analysis shows that MAGNET preserves trend and detail-band energy near the target level, whereas the compared LSTM attenuates high- and medium-frequency detail bands.
- Randomly permuting TGSM gates breaks trend-seasonal temporal alignment and degrades numerical metrics on all four financial datasets.
- Loss-decile analysis separates the piecewise-constant forward MSAL score from its smooth straight-through surrogate gradient.

## Complexity and Sensitivity

Under the common CPU evaluation environment, MAGNET uses about 611 MB of incremental peak memory and about 11.41 seconds per epoch, ranking sixth of 14 models in per-epoch training time. This is lower memory than Peri-midFormer, TimeMixer, and Timer-XL, although simpler baselines remain lighter.

One-factor-at-a-time sensitivity analyses cover the decomposition-kernel size `k_d`, the CASR hidden dimension `h`, and the MSAL weight `lambda`. Results remain broadly stable over moderate ranges of each hyperparameter. Removing MSE entirely (`lambda = 1`) degrades RMSE and directional accuracy, consistent with MSAL acting as a surrogate-augmented sign-alignment term rather than as a standalone point-loss replacement.

## Implementation Notes

The reported experiments use the following default settings: input length 10, forecast horizon 1, hidden dimension 64, Legendre order 64, decomposition kernel 5, CASR hidden dimension 32, MSAL weight 0.5, dropout 0.1, batch size 16, and 100 training epochs with AdamW and cosine-annealing learning-rate scheduling from 1e-4. The random seed is fixed at 42. Sensitivity configurations are repeated three times and reported as means. Baselines follow their published settings with dataset-specific adaptation.

## Repository Layout

```text
MAGNET/
|-- dataset/                     raw market and auxiliary benchmark data
|-- model/                       MAGNET implementation
|-- main.py                      main training and evaluation script
`-- README.md
```

`main.py` and `model/Ours_MAGNET/transformer5.7-NDX.py` follow the manuscript protocol: chronological train/validation/test construction, training-set-only standardization, the dual-branch encoder, trend-guided fusion, and the hybrid MSE-MSAL objective. Data paths and hyperparameters are declared at the top of each script.
