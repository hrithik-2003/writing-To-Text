import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

import sys

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
    # Create an image data generator with desired augmentation options
    datagen = ImageDataGenerator(
        rotation_range=5,  # Randomly rotate the images by up to 10 degrees
        width_shift_range=0.15,  # Randomly shift the images horizontally by up to 10% of the width
        height_shift_range=0.15,  # Randomly shift the images vertically by up to 10% of the height
        zoom_range=0.1,  # Randomly zoom the images by up to 10%
    )

    # Expand dimensions to match expected input shape for the CNN model
    images = np.expand_dims(images, axis=-1)

    # Perform data augmentation
    augmented_images = []
    for image in images:
        augmented_image = datagen.random_transform(image)
        augmented_images.append(augmented_image)

    # Convert augmented images list to numpy array
    augmented_images = np.array(augmented_images)

    # Normalize pixel values to the range [0, 1]
    augmented_images = augmented_images / 255.0

    return augmented_images

# Define the CNN model architecture
def create_model(optimizer='adam', activation='relu', dropout_rate=0.25):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=activation, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(Flatten())
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():

    if os.path.exists('best_model.h5'):
        # Load the saved model
        best_model = load_model('best_model.h5')
        print("Loaded existing model.")
    else:

        # Load the MNIST images and labels
        train_images = load_mnist_images(train_images_file)
        test_images = load_mnist_images(test_images_file)
        train_labels = load_mnist_labels(train_labels_file)
        test_labels = load_mnist_labels(test_labels_file)

        # Preprocess the images
        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)

        # Split the dataset into training and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42)

        # Reshape the images for compatibility with CNN model
        train_images = train_images.reshape((-1, 28, 28, 1))
        val_images = val_images.reshape((-1, 28, 28, 1))
        test_images = test_images.reshape((-1, 28, 28, 1))

        # Create the KerasClassifier wrapper for scikit-learn
        keras_model = KerasClassifier(build_fn=create_model, verbose=0)

        # Define the hyperparameters to tune
        parameters = {
            'optimizer': ['adam', 'rmsprop'],
            'activation': ['relu', 'sigmoid']
        }

        # Perform grid search using cross-validation
        print("Selecting model...")
        sys.stdout.flush()  # Flush the buffer to immediately display the message
        grid_search = GridSearchCV(estimator=keras_model, param_grid=parameters, cv=3)
        grid_search.fit(train_images, train_labels)

        # Print the best parameters and best accuracy
        print("Model selected.")
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: ", grid_search.best_score_)

        # Get the best model
        best_model = grid_search.best_estimator_.model

        # Train the best model on the full training set
        best_model.fit(train_images, train_labels, epochs=10)

        # Save the best model
        best_model.save('best_model.h5')

def evaluate_model():
    # Load the MNIST images and labels
    test_images = load_mnist_images(test_images_file)
    test_labels = load_mnist_labels(test_labels_file)

    # Preprocess the images
    test_images = preprocess_images(test_images)

    # Reshape the images for compatibility with CNN model
    test_images = test_images.reshape((-1, 28, 28, 1))

    # Load the saved best model
    best_model = load_model('best_model.h5')

    # Evaluate the best model on the test set
    test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Make predictions on the test set
    test_predictions = np.argmax(best_model.predict(test_images), axis=-1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# Run the model training
if __name__ == '__main__':
    train_model()  # Call the train_model function to train and save the best model
    evaluate_model()  # Call the evaluate_model function to load the saved model and perform evaluations
