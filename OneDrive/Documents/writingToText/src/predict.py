import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from PIL import Image


# Load the trained model
best_model = load_model('best_model.h5')

# Provide the path to the image file
current_dir = os.getcwd()
image_path = os.path.join(current_dir, 'images', 'img_5.jpg')

# Function to preprocess the image
def preprocess_image(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match the model's input shape
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values (same as during training)
    input_data = np.expand_dims(img_array, axis=0)  # Reshape the image array
    return input_data

def center_digit_image(digit_image, digit_width):
    image_width, image_height = digit_image.size

    if image_width < digit_width:
        # Calculate the number of pixels to add on each side
        num_pixels_to_add = digit_width - image_width
        num_pixels_to_add_left = num_pixels_to_add // 2
        num_pixels_to_add_right = num_pixels_to_add - num_pixels_to_add_left

        # Create a new blank image with the required width
        new_image = Image.new("L", (digit_width, image_height), color=0)

        # Paste the digit image in the center with black background
        new_image.paste(digit_image, (num_pixels_to_add_left, 0))

        return new_image

    return digit_image



# Function to split the image into individual digits
def split_digits(image):
    digit_images = []  # List to store individual digit images
    width, height = image.size
    digit_width = width // num_digits  # Assuming all digits have equal width

    for i in range(num_digits):
        x1 = i * digit_width
        y1 = 0
        x2 = (i + 1) * digit_width
        y2 = height
        digit_image = image.crop((x1, y1, x2, y2))

        digit_image = center_digit_image(digit_image, digit_width)
        
        digit_images.append(digit_image)

    return digit_images

# Function to get the predicted labels for a set of digits
def get_predicted_labels(digit_images):
    predicted_labels = []
    for digit_image in digit_images:
        input_data = preprocess_image(digit_image)
        predictions = best_model.predict(input_data)
        predicted_label = np.argmax(predictions, axis=-1)
        predicted_label = int(predicted_label[0])  # Extract the integer value
        predicted_labels.append(predicted_label)

        digit_image.show()
    return predicted_labels

# Load the image
image = Image.open(image_path)

# Specify the number of digits in the image
num_digits = 11

# Split the image into individual digit images
digit_images = split_digits(image)

# Get the predicted labels for each digit
predicted_labels = get_predicted_labels(digit_images)

print("Predicted Labels:", predicted_labels)
