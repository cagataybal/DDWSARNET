DDWSARNET: Layer-Wise Dynamic-τ WSAR Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official MATLAB implementation of DDWSARNET.**

This repository contains the source code for the paper:  
**"DDWSARNET: A Layer-Wise Dynamic-τ WSAR Framework for Probabilistic Calibration and Noise Robustness in Deep Neural Networks"**.

## 📌 Overview

**DDWSARNET** is a gradient-free training framework designed to solve the "overfitting paradox" of standard Backpropagation in noisy environments. 
By decomposing the high-dimensional optimization problem into **layer-wise block-coordinate sub-problems** and utilizing a **Dynamic-τ  Weighted Superposition Attraction-Repulsion (WSAR)** metaheuristic, DDWSARNET achieves superior probabilistic calibration (minimizing Log-Loss) compared to SGD/Adam and monolithic metaheuristics (PSO, WSAR).

### Key Features
* **Layer-Wise Optimization:** Optimizes weights layer-by-layer to mitigate the curse of dimensionality.
* **Dynamic-τ  Mechanism:** A bi-directional controller that automatically balances Exploration (low τ) and Exploitation (high τ) based on loss landscape stagnation.
* **Elite-Mix Initialization:** Uses robust statistics (IQR-based clipping) from a global Elite Archive to warm-start block optimizations.
* **Noise Robustness:** Proven resilience against Feature Noise (AWGN) and Symmetric Label Flipping.

## 📂 Repository Structure

```text
DDWSARNET/
├── main.m                     # Example script to run the benchmarks (Two-Moons / Diabetes)
├── wsar_deep_layerwise_*.m    # Main wrapper functions for the experiments
├── algorithms/
│   ├── wsar.m                 # The core WSAR metaheuristic engine
│   ├── wsar_layerwise.m       # The proposed layer-wise block coordinate logic
│   └── wsarset.m              # Configuration object for WSAR parameters
├── utils/
│   ├── make_moons.m           # Synthetic dataset generator
│   ├── nn_ce_single.m         # Cross-Entropy Loss calculations
│   └── layer_slices.m         # Helper for parameter decomposition
├── LICENSE                    # MIT License file
└── README.md                  # Project documentation
```

## 🚀 Installation & Requirements

This code is implemented in MATLAB.
No external compilation is required.
Prerequisites:MATLAB (R2020b or newer recommended)
Deep Learning Toolbox (Required only for the CNN comparison baseline)
Statistics and Machine Learning Toolbox (Required for particleswarm baseline comparison)

Setup:Clone the repository:
git clone [https://github.com/cagataybal/DDWSARNET.git](https://github.com/cagataybal/DDWSARNET.git)
cd DDWSARNET
Add the folders to your MATLAB path.

## 💻 Usage

To reproduce the experimental results (e.g., Two-Moons benchmark with noise), you can run the main configuration script.

Example: Running a Robustness Test
Matlab Configuration: Feature Noise = 0.3, Label Flip Rate = 0.2
featNoise = 0.3;
flipRate  = 0.2;
trainRatio = 0.8;
hidden_sizes = [16, 16]; % 2-Hidden Layer MLP

Run the benchmark comparison (DDWSARNET vs. BP vs. PSO)
[opt, summary] = wsar_deep_layerwise_vs_bp_pso_cnn_cfg(featNoise, flipRate, trainRatio, hidden_sizes);

Display Results
disp('Test Cross-Entropy Results:');
fprintf('DDWSARNET: %.4f\n', summary.wsar_layerwise.ce);
fprintf('BP (Adam): %.4f\n', summary.bp.ce);

## 📉 Reproduction of Paper Results

The repository includes scripts to generate the "Win Rate" tables and noise robustness plots (Fig. 2 and Fig. 3 in the manuscript).
Run run_sensitivity_analysis.m (if available) to sweep through \sigma_{noise}\in\left[0.1,0.5\right].
Results will be saved in the results/ directory.

## 📜 Citation

If you use this code or the DDWSARNET algorithm in your research, please cite our paper:
Text:Bal, C.,. "DDWSARNET: A Layer-Wise Dynamic-τ WSAR Framework for Probabilistic Calibration and Noise Robustness in Deep Neural Networks." Under Review, 2026.
BibTeX:Code snippet@article{ddwsarnet2026,
  title={DDWSARNET: A Layer-Wise Dynamic-τ  WSAR Framework for Probabilistic Calibration and Noise Robustness in Deep Neural Networks},
  author={Bal, Cagatay},
  journal={Under Review},
  year={2026},
  note={Preprint}
}
⚖️ LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
***
