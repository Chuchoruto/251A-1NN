import torch
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_centroids(dataset):
    sums = {digit: None for digit in range(10)}
    counts = {digit: 0 for digit in range(10)}
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]  # img is shape (1, 28, 28)
        if sums[label] is None:
            sums[label] = img.clone()
        else:
            sums[label] += img
        counts[label] += 1
    
    centroids = {}
    for digit in range(10):
        centroids[digit] = sums[digit] / counts[digit]
    return centroids

def visualize_centroids(centroids, out_file):
    """
    centroids: dict {label: tensor of shape (1, 28, 28)}
    out_file: path (string) to save the figure, e.g. './plots/centroids.png'
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    axs = axs.flatten()
    
    for digit in range(10):
        centroid_img = centroids[digit].squeeze(0).numpy()  # shape (28, 28)
        axs[digit].imshow(centroid_img, cmap='gray')
        axs[digit].set_title(f"Digit {digit}")
        axs[digit].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_file)  # Save the figure to file
    plt.close(fig)         # Close the figure to free memory

def centroid_1nn_classify(test_img, centroids):
    """
    test_img: shape (1,28,28) or (784,)
    centroids: dict { label: centroid_tensor (1,28,28) }
    Returns the label of the nearest centroid by L2 distance.
    """
    # You can flatten or keep shape as (1,28,28), so long as you do consistent distance
    test_flat = test_img.view(-1)  # shape (784,)
    
    best_label = None
    best_dist = float('inf')
    for label, centroid in centroids.items():
        centroid_flat = centroid.view(-1)  # shape (784,)
        dist = torch.norm(test_flat - centroid_flat, p=2)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def main():
    # 1) Load MNIST
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 2) Compute 10 centroids (one per digit)
    print("Computing centroids...")
    centroids = compute_centroids(train_dataset)
    
    # 3) Visualize these centroids
    output_path = './plots/centroids.png'
    visualize_centroids(centroids, output_path)
    print(f"Centroid plot saved to: {output_path}")
    
    # 4) Evaluate on test set
    correct = 0
    total = 0
    
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        pred_label = centroid_1nn_classify(img, centroids)
        
        if pred_label == label:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"Accuracy using 10 centroids: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
