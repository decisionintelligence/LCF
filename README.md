# Latent Causal Flow: Continuous Backdoor Adjustment for Time Series Generation

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%2F%2012.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

## 🚀 Introduction

Deep generative models have achieved remarkable success in time series generation (TSG), yet they largely neglect interventional and counterfactual inference. Existing causal generative methods typically rely on discrete environment prototypes, which fail to capture the continuous nature of real-world confounding. 

To address this, we propose **Latent Causal Flow (LCF)**, a flow matching framework that models latent confounders as a continuous space with a Condition-Independent Latent Environment Prior:

*   🌐 **Continuous Backdoor Adjustment:** We provide theoretical justification proving that Pearl's backdoor adjustment corresponds to the expectation of velocity fields over a continuous latent distribution.
*   🔄 **Dual-Path Environment Injection:** An environment-aware velocity architecture that dynamically integrates latent environments into flow matching.
*   🧩 **Causal Pathway Disentanglement (CPD):** A mechanism that explicitly decomposes generation into condition-driven, environment-driven, and interaction-driven causal effects.
*   🎯 **Deterministic Counterfactuals:** Leveraging the deterministic invertibility of flow matching ODEs to achieve exact, individual-level counterfactual generation ("what-if" reasoning), overcoming the stochastic limitations of diffusion SDEs.
