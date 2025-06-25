import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Parameters
img_height, img_width = 80, 80  # Ensure these match your trained model's input size

# Load the trained model
model_path = r"C:\Users\USER\Downloads\ship_classifier (2).keras"  # Path to your saved model
model = load_model(model_path)

# Function to classify an image with confidence and overlay the result outside the image
def classify_and_overlay(image_path):
    """
    Classify whether a ship is present in the given image with confidence score
    and overlay the result outside the image.
    :param image_path: Path to the image to classify
    :return: The image with the overlay text outside
    """
    try:
        if not os.path.exists(image_path):
            return f"Error: Image file '{image_path}' not found."

        # Load and preprocess the image
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction
        prediction = model.predict(img_array)
        confidence = prediction[0][0]  # Confidence score for "Ship"

        # Interpret the result
        if confidence > 0.5:
            result_text = f"Ship detected ({confidence * 100:.2f}%)"
        else:
            result_text = f"No Ship detected ({(1 - confidence) * 100:.2f}%)"

        # Load the original image for overlay
        original_img = cv2.imread(image_path)

        # Create a blank area below the image for text
        padding = 50  # Height of the blank padding area
        blank_area = np.ones((padding, original_img.shape[1], 3), dtype=np.uint8) * 255  # White blank area
        combined_img = np.vstack((original_img, blank_area))  # Combine image and blank area

        # Choose font and text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Smaller font scale for text
        thickness = 2
        color = (0, 0, 255)  # Red text

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(result_text, font, font_scale, thickness)

        # Calculate the position to center the text within the blank area
        text_x = max(10, (combined_img.shape[1] - text_width) // 2)  # Center horizontally
        text_y = original_img.shape[0] + (padding // 2) + (text_height // 2)  # Position in the blank area

        # Add text overlay in the blank area
        cv2.putText(combined_img, result_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Convert BGR to RGB for displaying with Matplotlib
        combined_img_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

        # Use Matplotlib to display the image with overlay
        plt.imshow(combined_img_rgb)
        plt.axis('off')  # Remove axis for cleaner display
        plt.show()

        return result_text

    except Exception as e:
        return f"Error during image classification: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Path to the image you want to classify
    image_path = r"C:\Users\USER\Downloads\real time no ship detection\397293cb9.jpg" # Replace with your test image path

    # Perform classification and overlay result on image
    result = classify_and_overlay(image_path)
    print(result)








