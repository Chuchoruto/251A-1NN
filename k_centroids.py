import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import numpy as np
import time

def flatten_img(img):
    # (1, 28, 28) -> (784,)
    return img.view(-1)

def main():
    # --- 1) Load MNIST from torchvision (CPU) ---
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    print(f"Train samples: {len(train_dataset)}")  # 60000
    print(f"Test samples: {len(test_dataset)}\n")  # 10000
    
    # --- 2) Separate the training data by digit for k-means ---
    # We'll store them in lists, keyed by digit
    digit_to_images = {d: [] for d in range(10)}
    
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        digit_to_images[label].append(flatten_img(img).numpy())  # store as NumPy
    
    # Convert lists to 2D NumPy arrays
    for d in range(10):
        digit_to_images[d] = np.vstack(digit_to_images[d])  # shape: (num_samples_for_digit_d, 784)
    
    # --- 3) Run k-means for each digit class ---
    k = 150  # e.g., 5 centroids per class
    print(f"Computing {k} centroids per digit using k-means...")
    
    digit_to_centroids = {}  # digit -> shape (k, 784)
    
    for d in range(10):
        X_d = digit_to_images[d]
        print(f"  Digit {d}: {X_d.shape[0]} samples -> k-means(k={k})")
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_d)
        
        # kmeans.cluster_centers_ has shape (k, 784)
        digit_to_centroids[d] = kmeans.cluster_centers_
    
    # --- 4) Build a single matrix of all centroids (10*k, 784) and their labels (10*k,)
    all_centroids = []
    all_labels = []
    
    for d in range(10):
        c = digit_to_centroids[d]  # shape (k, 784)
        all_centroids.append(c)
        # assign label d to each centroid in c
        all_labels.extend([d]*c.shape[0])
    
    all_centroids = np.vstack(all_centroids)  # shape: (10*k, 784)
    all_labels = np.array(all_labels, dtype=np.int64)  # shape: (10*k,)
    
    print(f"\nTotal centroids: {all_centroids.shape[0]} (for 10 digits * k={k})")

    # --- 5) Classify test set by nearest centroid ---
    # We'll do an L2 distance. For speed, let's use the dot-product trick again:
    #
    #  ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x.dot(c)
    #
    #  But since we only have 10*k centroids, a direct loop over them is also feasible.
    #
    #  However, let's do a vectorized approach in NumPy for demonstration.

    # Precompute squared norms of centroids (shape: (10*k,))
    centroid_sq_norms = (all_centroids ** 2).sum(axis=1)

    # We'll do the test set in small batches or all at once (10,000 test samples with 10*k=50 means 500k distances).
    # 500k is not huge, so we can probably do it in one shot. But let's be safe and do small batches if needed.

    test_batch_size = 1000
    correct = 0
    total = 0

    print("\nClassifying test set...")
    start_time = time.time()

    for start_idx in range(0, len(test_dataset), test_batch_size):
        end_idx = min(start_idx + test_batch_size, len(test_dataset))
        batch_size = end_idx - start_idx
        
        # Build an array of shape (batch_size, 784)
        test_batch = []
        test_labels_batch = []
        for i in range(start_idx, end_idx):
            img, lab = test_dataset[i]
            test_batch.append(flatten_img(img).numpy())
            test_labels_batch.append(lab)
        
        test_batch = np.vstack(test_batch)  # shape (batch_size, 784)
        test_labels_batch = np.array(test_labels_batch, dtype=np.int64)
        
        # Compute squared norms for test batch (batch_size,)
        test_sq_norms = (test_batch ** 2).sum(axis=1)
        
        # Dot products: (batch_size, 784) @ (784, 10*k) -> (batch_size, 10*k)
        dot_products = test_batch @ all_centroids.T
        
        # dist_matrix[i, j] = test_sq_norms[i] + centroid_sq_norms[j] - 2*dot_products[i, j]
        # shape: (batch_size, 10*k)
        dist_matrix = (
            test_sq_norms.reshape(-1, 1) 
            + centroid_sq_norms.reshape(1, -1) 
            - 2.0 * dot_products
        )
        
        # Argmin over dimension 1 (the centroid dimension)
        nearest_indices = dist_matrix.argmin(axis=1)  # shape (batch_size,)
        predicted_labels_batch = all_labels[nearest_indices]
        
        correct += np.sum(predicted_labels_batch == test_labels_batch)
        total += batch_size

    end_time = time.time()
    accuracy = correct / total * 100

    print(f"\nDone! Accuracy with {k} centroids per digit = {accuracy:.2f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
