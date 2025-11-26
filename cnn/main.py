import tkinter as tk
import numpy as np
from tensorflow import keras
from gui_app import HandwritingRecognitionApp

# ---------------------------------------------------------
# 1. LOAD THE TRAINED MODEL
# ---------------------------------------------------------
print("Loading the AI model...")
try:
    # We load the CNN model we trained earlier
    model = keras.models.load_model("models/cnn_model.h5")
    print("Model loaded successfully!")
except:
    print("Error: Could not load the model. Make sure 'cnn_model.h5' is in the 'models' folder.")
    model = None

# ---------------------------------------------------------
# 2. DEFINE WHAT HAPPENS WHEN WE PREDICT
# ---------------------------------------------------------
def predict_digit(image_array):
    """
    This function runs when you click the 'Predict' button.
    It receives the drawing as a number array.
    """
    # Check if model is loaded
    if model is None:
        return "Error", "Model is missing!"

    # 1. Reshape the image for the model (1 image, 28x28 pixels, 1 color channel)
    input_data = image_array.reshape(1, 28, 28, 1)

    # 2. Ask the model to predict
    # It returns a list of probabilities for each digit (0-9)
    predictions = model.predict(input_data, verbose=0)[0]

    # 3. Find the digit with the highest probability
    predicted_digit = np.argmax(predictions)
    
    # 4. Get the confidence score (how sure is the model?)
    confidence_score = predictions[predicted_digit] * 100

    print(f"I think you drew a: {predicted_digit} ({confidence_score:.1f}% sure)")

    # Return the result to the GUI so it can display it
    result_text = str(predicted_digit)
    confidence_text = f"Confidence: {confidence_score:.1f}%"
    
    return result_text, confidence_text

# ---------------------------------------------------------
# 3. START THE APPLICATION
# ---------------------------------------------------------
def main():
    # Create the main window
    root = tk.Tk()

    # Define the function that connects the GUI to our prediction logic
    def on_gui_predict(image_array):
        # 1. Get the prediction from our model
        digit, confidence = predict_digit(image_array)
        
        # 2. Update the GUI with the result
        app.update_result(digit, confidence)

    # Start the App
    # We pass our function so the GUI can send us the image
    app = HandwritingRecognitionApp(root, prediction_callback=on_gui_predict)

    print("Starting application...")
    root.mainloop()

if __name__ == "__main__":
    main()
