import torch
import torchvision
import torchvision.transforms as transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # --- 1) Load the entire MNIST train & test sets ---
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    print(f"Training samples: {len(train_dataset)}")  # 60,000
    print(f"Test samples: {len(test_dataset)}\n")     # 10,000

    # --- 2) Flatten & move training data to GPU ---
    print("Preparing training data...")
    train_size = len(train_dataset)
    
    train_images_list = []
    train_labels_list = []
    for i in range(train_size):
        img, label = train_dataset[i]
        # Flatten (1, 28, 28) -> (784,)
        train_images_list.append(img.view(-1))
        train_labels_list.append(label)
    
    train_images = torch.stack(train_images_list).to(device)  # (train_size, 784)
    train_labels = torch.tensor(train_labels_list, dtype=torch.long, device=device)
    
    del train_images_list, train_labels_list  # free some CPU memory
    
    print(f"train_images shape = {train_images.shape}, dtype = {train_images.dtype}")
    print(f"train_labels shape = {train_labels.shape}, dtype = {train_labels.dtype}")

    # --- 3) Flatten test data (remain on CPU for now) ---
    print("\nPreparing test data...")
    test_size = len(test_dataset)
    
    test_images_list = []
    test_labels_list = []
    for i in range(test_size):
        img, label = test_dataset[i]
        test_images_list.append(img.view(-1))  # (784,)
        test_labels_list.append(label)
    
    # We'll move test images to GPU in batches
    test_images = torch.stack(test_images_list)         # (test_size, 784), still on CPU
    test_labels = torch.tensor(test_labels_list, dtype=torch.long)  # on CPU

    print(f"test_images shape = {test_images.shape}, dtype = {test_images.dtype}")
    print(f"test_labels shape = {test_labels.shape}, dtype = {test_labels.dtype}")

    # --- 4) To speed up distance calc, precompute squared norms of training images ---
    # This is just (train_size,)  i.e. sum of squares per row
    train_sq_norms = train_images.pow(2).sum(dim=1)  # still on GPU, shape: (train_size,)

    # --- 5) 1-NN classification in batches to limit GPU usage ---
    test_batch_size = 2000  # Adjust this to fit your GPU memory (e.g. 1000, 2000, etc.)
    correct = 0
    total = 0

    print(f"\nStarting 1-NN in batches of size {test_batch_size}...")
    start_time = time.time()

    for start_idx in range(0, test_size, test_batch_size):
        end_idx = min(start_idx + test_batch_size, test_size)
        current_batch_size = end_idx - start_idx

        # Move current test batch to GPU
        test_batch = test_images[start_idx:end_idx].to(device)  # (batch_size, 784)
        test_labels_batch = test_labels[start_idx:end_idx].to(device)

        # 1) Compute squared norms of test batch: shape (batch_size,)
        test_sq_norms = test_batch.pow(2).sum(dim=1)  # (batch_size,)

        # 2) Compute dot product: (batch_size, 784) @ (784, train_size) -> (batch_size, train_size)
        dot_products = test_batch @ train_images.T

        # 3) Expand norms to match (batch_size, train_size):
        #    test_sq_norms -> (batch_size, 1)
        #    train_sq_norms -> (1, train_size)
        # Then dist_matrix[i, j] = test_sq_norms[i] + train_sq_norms[j] - 2*dot_products[i, j]
        dist_matrix = (
            test_sq_norms.unsqueeze(1) 
            + train_sq_norms.unsqueeze(0)
            - 2.0 * dot_products
        )
        # shape: (batch_size, train_size)

        # 4) Argmin over train_size dimension to find the nearest neighbor
        _, nearest_indices = dist_matrix.min(dim=1)
        predicted_labels_batch = train_labels[nearest_indices]

        # Count how many are correct in this batch
        correct_batch = (predicted_labels_batch == test_labels_batch).sum().item()
        correct += correct_batch
        total += current_batch_size

        # Free memory for next batch
        del dist_matrix, dot_products, test_batch, test_labels_batch, test_sq_norms

    end_time = time.time()
    accuracy = correct / total * 100

    print(f"\nDone! Accuracy on full test set: {accuracy:.2f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
