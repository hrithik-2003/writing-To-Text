import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Defining folder path
dataset_folder = 'Dataset'
train_images_file = os.path.join(dataset_folder, 'train-images.idx3-ubyte')
test_images_file = os.path.join(dataset_folder, 't10k-images.idx3-ubyte')
train_labels_file = os.path.join(dataset_folder, 'train-labels.idx1-ubyte')
test_labels_file = os.path.join(dataset_folder, 't10k-labels.idx1-ubyte')

# Load the MNIST dataset
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and the number of images
        magic_num = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')

        # Read the dimensions of the images
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Read the pixel values of the images
        images = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape the pixel values into a 3D array
        images = images.reshape((num_images, num_rows, num_cols))

        return images

# Load the MNIST labels
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and the number of labels
        magic_num = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read the labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

# Preprocess the images
def preprocess_images(images):
    # Normalize pixel values to the range [0, 1]
    images = images / 255.0

    # Additional preprocessing steps can be added here if required

    return images

# Print the first 3 images from the dataset
def print_images(images):
    for i in range(3):
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.show()

# Main function for verifying the code
def verify_code():
    # Load the MNIST train images
    train_images = load_mnist_images(train_images_file)
    print("Train Images:")
    print_images(train_images[:3])

    # Load the MNIST train labels
    train_labels = load_mnist_labels(train_labels_file)

    # Load the MNIST test images
    test_images = load_mnist_images(test_images_file)
    print("Test Images:")
    print_images(test_images[:3])

    # Split the dataset into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)
    
    

    # Print the shapes of the split datasets
    print("Train Images shape:", train_images.shape)
    print("Train Labels shape:", train_labels.shape)
    print("Validation Images shape:", val_images.shape)
    print("Validation Labels shape:", val_labels.shape)
    print("Test Images shape:", test_images.shape)
    print('Val Images : ')
    print_images(val_images[:3])
    print('val_labels', val_labels[:3])

def preprocess_data():
    # Load the MNIST images
    train_images = load_mnist_images(train_images_file)
    test_images = load_mnist_images(test_images_file)

    # Preprocess the images
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    # Load the MNIST labels
    train_labels = load_mnist_labels(train_labels_file)

    # Split the dataset into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)

    # Save the preprocessed images or perform any additional processing

# Run the data preprocessing
if __name__ == '__main__':
    verify_code()
    preprocess_data()
