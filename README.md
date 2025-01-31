# MNIST 1-Nearest Neighbor Experiment

This document provides instructions for setting up the environment, running experiments, and analyzing results. It also explains the methodology for selecting prototypes and compares performance between **random sampling** and **k-means centroids** when keeping the total number of samples constant.

---

## 1. Setting Up the Environment

To ensure reproducibility and maintain dependencies, it's recommended to use a **virtual environment**.

### **Creating a Virtual Environment**
Run the following command to create and activate a virtual environment:

```bash
# Create a virtual environment named 'mnist_env'
python3 -m venv mnist_env

# Activate the virtual environment
# On macOS/Linux:
source mnist_env/bin/activate

# On Windows (Command Prompt):
mnist_env\Scripts\activate
```

### **Installing Dependencies**
Ensure you have `pip` installed and then install all required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install dependencies such as `numpy`, `torch`, `torchvision`, `scikit-learn`, and `matplotlib`

---

## 2. Running the Experiments

### **Running K-Means and Random Sampling Experiments**
To run the experiment multiple times and save results:

```bash
python run_experiments.py
```

This will perform **10 repeated runs** for each experimental setting and save results into CSV files in the `./results/` folder.

### **Analyzing the Results**
To compute the **mean and standard deviation** for accuracy and classification time:

```bash
python analyze_results.py
```

This script will summarize the performance of each experiment and print results to the terminal.

---

## 3. Methodology: Prototype Selection

The experiments explore two ways to select **prototypes** for the 1-Nearest Neighbor classification:

### **(A) K-Means Centroids Per Class**
- The dataset is **divided by digit** (0–9).
- **K-Means clustering** is applied separately to each digit.
- The cluster centroids serve as the **prototypes**.
- Example: If `k=100`, this results in **1,000 total prototypes** (100 per class × 10 classes).
- These prototypes are well-distributed representations of each digit class.

### **(B) Randomly Sampled Training Data**
- Instead of clustering, we **randomly sample** a fixed number of training images.
- These randomly chosen images serve as **prototypes**.
- Example: If `m=1000`, we select 1,000 images directly from the training set.
- These prototypes may not fully capture the distribution of each class, but is used a baseline comparison

---

## 4. Performance Comparison: K-Means vs. Random Sampling

We compare **classification accuracy** and **runtime** across different prototype selection methods while keeping the total number of samples constant.

| Number of Samples | K-Means Centroids Accuracy (%) | Random Sampling Accuracy (%) |
|-------------------|--------------------------------|----------------------------|
| 1000              |     95.76 ± 0.13               |       88.72 ± 0.33         |
| 5000              |     96.76 ± 0.08               |       93.59 ± 0.21         |
| 10000             |     96.91 ± 0.12               |       94.84 ± 0.20         |

**Observations:**
- K-Means centroids generally perform **better** than random sampling because they summarize important variations in each class.
- Random sampling can sometimes work well, but the accuracy depends on whether important digit variations are included in the sample.
- **Runtime** for classification is similar for both methods, as both involve comparing test images to the same number of prototypes.

---

## 5. Baseline Performance (Full Dataset 1-NN)

To evaluate the baseline **1-Nearest Neighbor accuracy** using **all 60,000 training samples**, run:

```bash
python3 1nn_full_batched.py
```

This script performs **full** 1-NN classification, where every test image is compared against all 60,000 training samples.

**Baseline Accuracy:** 96.91%

**Trade-off:**
- **High accuracy** but **slower** due to the large number of distance computations

---

## 6. Summary

- **K-Means prototypes** provide a structured way to condense the dataset while retaining high accuracy.
- **Random sampling** is easier but may lead to lower accuracy due to uneven class representation.
- The **baseline 1-NN (full dataset)** achieves the best accuracy but is computationally expensive.
- The scripts allow for systematic experimentation, data collection, and statistical analysis.

For further modifications, adjust `k_values` and `m_values` in `run_experiments.py` to explore different prototype sizes.

---

## References
- Yann LeCun et al., "Gradient-Based Learning Applied to Document Recognition," Proceedings of the IEEE, 1998.
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/

