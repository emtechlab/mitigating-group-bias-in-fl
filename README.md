# Mitigating Group Bias in Federated Learning for Heterogeneous Devices

[![ArXiv](https://img.shields.io/badge/arXiv-2309.07085v2-blue.svg)](https://arxiv.org/abs/2309.07085v2)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A privacy-preserving, group-fair federated learning framework that mitigates bias arising from feature heterogeneity across edge devices by leveraging multiplicative-weights updates with regularization and performance thresholds.  

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Installation & Prerequisites](#installation--prerequisites)  
3. [Data Preparation](#data-preparation)  
4. [Configuration](#configuration)  
5. [Usage](#usage)  
   - [Training](#training)   
6. [Experiments & Results](#experiments--results)  
7. [Citation](#citation)  

---

## Introduction

Federated learning enables decentralized model training on heterogeneous edge devices without sharing raw data, preserving user privacy while aggregating local updates into a global model. However, variation in device quality and sensing environments introduces **feature heterogeneity**, causing global models to underperform on groups with noisier data and perpetuate bias.  

We introduce a **Multiplicative Weights update with Regularization (MWR)** framework that:  
- Computes **privacy-preserving group importance weights** via average conditional probabilities across clients.  
- Applies a **modified multiplicative-weights algorithm** with an L1 regularizer to prevent weight explosion and improve worst-group performance.  
- Enforces a **performance threshold** on the best-performing group to avoid degrading high-accuracy groups.  
- Demonstrates superior worst-group true-positive-rate gains (up to +41%) on CIFAR-10, MNIST, Fashion-MNIST, USPS, SynthDigits, and MNIST-M without significant loss in overall accuracy.  

---

## Installation & Prerequisites

1. **Clone the repository**  
   ```bash
   git clone https://github.com/emtechlab/mitigating-group-bias-in-fl.git
   cd mitigating-group-bias-in-fl

2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # on Linux/macOS

3. **Install required packages**
   ```bash
   pip install -r requirements.txt

## Usage