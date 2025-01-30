#!/usr/bin/env python

import os
import time
import csv
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torch.utils.data import Subset, random_split

###############################################################################
#                           Data Preparation Functions                        #
###############################################################################

def load_mnist(root="./data"):
    """
    Loads the full MNIST train and test datasets (on CPU) with transforms.ToTensor().
    Returns:
        train_dataset, test_dataset
    """
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset

def get_digit_to_images(train_dataset):
    """
    Groups all training images by digit label in a dict: { digit: [flattened_images...] }.
    Each flattened image is a NumPy array of shape (784,).
    Returns:
        digit_to_images: {digit: np.array of shape (num_samples_for_digit, 784)}
    """
    digit_to_images_list = {d: [] for d in range(10)}
    
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        # Flatten (1,28,28) -> (784,)
        flat = img.view(-1).numpy()
        digit_to_images_list[label].append(flat)
    
    # Convert to 2D NumPy arrays
    digit_to_images = {}
    for d in range(10):
        digit_to_images[d] = np.vstack(digit_to_images_list[d])
    
    return digit_to_images

###############################################################################
#                     K-Means Prototypes (k centroids/class)                  #
###############################################################################

def compute_kmeans_prototypes(digit_to_images, k):
    """
    Runs k-means on each digit subset to produce k centroids per digit.
    Args:
        digit_to_images: { digit: np.array of shape (num_samples_for_digit, 784) }
        k: number of clusters per digit
    Returns:
        all_centroids: np.array of shape (10*k, 784)
        all_labels:    np.array of shape (10*k,) with each row's label
    """
    digit_to_centroids = {}
    for d in range(10):
        X_d = digit_to_images[d]
        
        # KMeans with no fixed seed => random initialization each time
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_d)
        
        digit_to_centroids[d] = kmeans.cluster_centers_  # shape (k, 784)
    
    # Combine into a single matrix
    all_centroids = []
    all_labels = []
    for d in range(10):
        c = digit_to_centroids[d]  # shape (k, 784)
        all_centroids.append(c)
        all_labels.extend([d] * c.shape[0])
    
    all_centroids = np.vstack(all_centroids)  # (10*k, 784)
    all_labels = np.array(all_labels, dtype=np.int64)  # (10*k,)
    return all_centroids, all_labels

###############################################################################
#                     Random Sampling Prototypes (m samples)                  #
###############################################################################

def sample_random_prototypes(train_dataset, m):
    """
    Randomly samples m images (and their labels) from the full training set.
    Returns:
        sample_vectors: np.array (m, 784)
        sample_labels:  np.array (m,)
    """
    if m > len(train_dataset):
        raise ValueError(f"m={m} cannot exceed total training samples {len(train_dataset)}")
    
    # Use random_split for convenience
    subset_m, _ = random_split(train_dataset, [m, len(train_dataset) - m])
    
    vectors_list = []
    labels_list = []
    for i in range(len(subset_m)):
        img, lab = subset_m[i]
        vectors_list.append(img.view(-1).numpy())  # shape (784,)
        labels_list.append(lab)
    
    sample_vectors = np.vstack(vectors_list)  # (m, 784)
    sample_labels = np.array(labels_list, dtype=np.int64)  # (m,)
    return sample_vectors, sample_labels

###############################################################################
#                           1-NN Classification                               #
###############################################################################

def classify_1nn(train_vectors, train_labels, test_dataset, batch_size=1000):
    """
    Performs 1-NN classification of the entire test_dataset using the given
    train_vectors, train_labels. Both are NumPy arrays:
        train_vectors: shape (N, 784)
        train_labels:  shape (N,)
    Returns:
        (accuracy, total_classify_time)
    """
    # Precompute squared norms of training vectors
    train_sq_norms = (train_vectors ** 2).sum(axis=1)  # shape (N,)
    
    correct = 0
    total = 0
    
    start_time = time.time()
    
    # We'll do the test set in batches
    test_size = len(test_dataset)
    for start_idx in range(0, test_size, batch_size):
        end_idx = min(start_idx + batch_size, test_size)
        current_batch_size = end_idx - start_idx
        
        # Build test batch
        test_batch = []
        test_labels_batch = []
        for i in range(start_idx, end_idx):
            img, lab = test_dataset[i]
            test_batch.append(img.view(-1).numpy())
            test_labels_batch.append(lab)
        
        test_batch = np.vstack(test_batch)            # shape (batch_size, 784)
        test_labels_batch = np.array(test_labels_batch, dtype=np.int64)
        
        # Dot products
        test_sq_norms = (test_batch ** 2).sum(axis=1)  # (batch_size,)
        dot_products = test_batch @ train_vectors.T    # (batch_size, N)
        
        # dist_matrix[i, j] = test_sq_norms[i] + train_sq_norms[j] - 2*dot_products[i, j]
        dist_matrix = (
            test_sq_norms.reshape(-1, 1) +
            train_sq_norms.reshape(1, -1) -
            2.0 * dot_products
        )
        
        nearest_indices = dist_matrix.argmin(axis=1)  # (batch_size,)
        predicted_labels = train_labels[nearest_indices]
        
        correct += np.sum(predicted_labels == test_labels_batch)
        total += current_batch_size
    
    end_time = time.time()
    accuracy = correct / total * 100.0
    classify_time = end_time - start_time
    
    return accuracy, classify_time

###############################################################################
#                              Main Experiment                                #
###############################################################################

def main():
    # Ensure results directory exists
    os.makedirs("./results", exist_ok=True)

    # 1) Load full MNIST once
    train_dataset, test_dataset = load_mnist()
    print(f"Loaded MNIST: {len(train_dataset)} train samples, {len(test_dataset)} test samples.\n")
    
    # 2) Pre-group the training data by digit for k-means usage
    digit_to_images = get_digit_to_images(train_dataset)
    
    # Lists to iterate over
    k_values = [100, 500, 1000]    # placeholder values for experiment
    m_values = [1000, 5000, 10000] # placeholder values for random sampling
    
    num_runs = 10  # We'll run each experiment setting 10 times
    
    # --------------------------------------------------------------------
    #            K-Means Prototypes (k centroids/class)
    # --------------------------------------------------------------------
    
    print("=================== K-Means Experiments ===================\n")
    for k in k_values:
        csv_filename = f"./results/k-means_{k}_centroids.csv"
        
        # Open the CSV file in write mode and add headers
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["accuracy", "classify_time"])
            
            # We run k-means -> 1-NN classification 10 times
            for run_idx in range(num_runs):
                print(f"[k={k}] Run {run_idx+1}/{num_runs} ...")
                
                # Compute the prototypes
                all_centroids, all_labels = compute_kmeans_prototypes(digit_to_images, k)
                
                # Classify test set
                accuracy, classify_time = classify_1nn(all_centroids, all_labels, test_dataset)
                
                # Write results to CSV
                writer.writerow([accuracy, classify_time])
                
        print(f"Saved results to {csv_filename}\n")
    
    # --------------------------------------------------------------------
    #         Random Sampling Prototypes (m samples)
    # --------------------------------------------------------------------
    
    print("================= Random Sampling Experiments ==============\n")
    for m in m_values:
        csv_filename = f"./results/random_sample_{m}.csv"
        
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["accuracy", "classify_time"])
            
            for run_idx in range(num_runs):
                print(f"[m={m}] Run {run_idx+1}/{num_runs} ...")
                
                # Randomly sample
                sample_vectors, sample_labels = sample_random_prototypes(train_dataset, m)
                
                # Classify test set
                accuracy, classify_time = classify_1nn(sample_vectors, sample_labels, test_dataset)
                
                writer.writerow([accuracy, classify_time])
                
        print(f"Saved results to {csv_filename}\n")


if __name__ == "__main__":
    main()
