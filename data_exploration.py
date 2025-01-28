import torch
import torchvision
import torchvision.transforms as transforms

# Define a transform to convert PIL images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download and load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',       # path where the data will be stored
    train=True,          # indicates we want the training set
    download=True,       # download it if it's not already there
    transform=transform  # apply the transform (to Tensor)
)

# Download and load the test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

def main():
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print()

    # Let's look at the shape of a single sample from the train_dataset
    image, label = train_dataset[0]
    print("Single sample shape (train_dataset[0][0]):", image.shape)
    print("Single sample label (train_dataset[0][1]):", label)
    print(f"Data type of the image: {image.dtype}")

    # If you’d like to see multiple samples in a row:
    for i in range(3):  # just look at the first 3 samples
        img, lbl = train_dataset[i]
        print(f"Sample {i} - Image shape: {img.shape}, Label: {lbl}")

if __name__ == "__main__":
    main()
