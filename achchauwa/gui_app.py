"""
Handwritten Character Recognition GUI
Using CNN model and Tkinter Canvas for drawing

This application allows users to draw digits/letters and get predictions
from the trained CNN model.
"""

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pickle
import os
import tensorflow as tf
from tensorflow import keras


class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Character Recognition")
        self.root.geometry("800x850")
        self.root.resizable(True, True)
        
        # Load the trained model
        self.model = None
        self.load_model()
        
        # Drawing variables
        self.canvas_size = 400
        self.brush_size = 20
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load the trained CNN model"""
        # Try CNN model first (recommended)
        cnn_model_path = "models/cnn_model.h5"
        knn_model_path = "models/knn_model.pkl"
        
        if os.path.exists(cnn_model_path):
            try:
                self.model = keras.models.load_model(cnn_model_path)
                self.model_type = 'CNN'
                print("CNN Model loaded successfully!")
                
                # Load metadata if available
                import json
                metadata_path = "models/cnn_model_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"Model metadata: {metadata}")
                return
            except Exception as e:
                print(f"Failed to load CNN model: {str(e)}")
        
        # Fallback to KNN model
        if os.path.exists(knn_model_path):
            try:
                with open(knn_model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_type = 'KNN'
                print("KNN Model loaded successfully!")
                
                # Load metadata if available
                metadata_path = "models/model_metadata.pkl"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    print(f"Model metadata: {metadata}")
                return
            except Exception as e:
                print(f"Failed to load KNN model: {str(e)}")
        
        # No model found
        messagebox.showerror(
            "Error", 
            f"Model file not found.\nPlease train the model first.\nLooking for: {cnn_model_path} or {knn_model_path}"
        )
        self.model_type = None
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Handwritten Character Recognition",
            font=("Arial", 24, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=(0, 20))
        
        # Canvas frame
        canvas_frame = tk.Frame(main_frame, bg="#f0f0f0")
        canvas_frame.pack()
        
        # Drawing canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            cursor="cross",
            bd=2,
            relief=tk.SOLID
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="Draw a digit (0-9) or letter (A-Z) in the box above",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#666666"
        )
        instructions.pack(pady=10)
        
        # Brush size control
        brush_frame = tk.Frame(main_frame, bg="#f0f0f0")
        brush_frame.pack(pady=10)
        
        tk.Label(
            brush_frame,
            text="Brush Size:",
            font=("Arial", 11),
            bg="#f0f0f0"
        ).pack(side=tk.LEFT, padx=5)
        
        self.brush_slider = tk.Scale(
            brush_frame,
            from_=10,
            to=40,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_brush_size,
            bg="#f0f0f0"
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # Predict button
        predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=30,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED
        )
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas,
            font=("Arial", 14, "bold"),
            bg="#f44336",
            fg="white",
            padx=30,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Result frame
        result_frame = tk.Frame(main_frame, bg="#f0f0f0")
        result_frame.pack(pady=20)
        
        tk.Label(
            result_frame,
            text="Prediction:",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        ).pack(side=tk.LEFT, padx=10)
        
        self.result_label = tk.Label(
            result_frame,
            text="--",
            font=("Arial", 32, "bold"),
            bg="white",
            fg="#2196F3",
            width=5,
            relief=tk.SOLID,
            bd=2
        )
        self.result_label.pack(side=tk.LEFT, padx=10)
        
        # Confidence label
        self.confidence_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 11),
            bg="#f0f0f0",
            fg="#666666"
        )
        self.confidence_label.pack()
    
    def update_brush_size(self, value):
        """Update brush size from slider"""
        self.brush_size = int(value)
    
    def start_draw(self, event):
        """Start drawing"""
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_on_canvas(self, event):
        """Draw on canvas with mouse"""
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill="black",
                width=self.brush_size
            )
            
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_draw(self, event):
        """Stop drawing"""
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="--")
        self.confidence_label.config(text="")
    
    def preprocess_image(self):
        """Preprocess the drawn image for model prediction"""
        # Invert colors (black on white to white on black)
        image = ImageOps.invert(self.image)
        
        # Get bounding box of the drawn content
        bbox = image.getbbox()
        
        if bbox is None:
            # Nothing drawn
            return None
        
        # Crop to bounding box with some padding
        padding = 20
        bbox = (
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(self.canvas_size, bbox[2] + padding),
            min(self.canvas_size, bbox[3] + padding)
        )
        image = image.crop(bbox)
        
        # Resize to 28x28 (MNIST size)
        image = image.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Return different shapes based on model type
        if self.model_type == 'CNN':
            # CNN expects (batch, height, width, channels)
            return image_array.reshape(1, 28, 28, 1)
        else:
            # KNN expects flattened (batch, features)
            return image_array.reshape(1, -1)
    
    def predict(self):
        """Make prediction on drawn image"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        # Preprocess image
        processed_image = self.preprocess_image()
        
        if processed_image is None:
            messagebox.showwarning("Warning", "Please draw something first!")
            return
        
        try:
            if self.model_type == 'CNN':
                # CNN model prediction
                probabilities = self.model.predict(processed_image, verbose=0)[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction] * 100
                
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_probs = probabilities[top_indices] * 100
                
                confidence_text = f"Confidence: {confidence:.1f}%\n"
                confidence_text += f"Top 3: {top_indices[0]}({top_probs[0]:.1f}%), "
                confidence_text += f"{top_indices[1]}({top_probs[1]:.1f}%), "
                confidence_text += f"{top_indices[2]}({top_probs[2]:.1f}%)"
                
                self.confidence_label.config(text=confidence_text)
                
            else:
                # KNN model prediction
                prediction = self.model.predict(processed_image)[0]
                
                # Get prediction probabilities (for confidence)
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(processed_image)[0]
                    confidence = max(probabilities) * 100
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(probabilities)[-3:][::-1]
                    top_classes = [self.model.classes_[i] for i in top_indices]
                    top_probs = [probabilities[i] * 100 for i in top_indices]
                    
                    confidence_text = f"Confidence: {confidence:.1f}%\n"
                    confidence_text += f"Top 3: {top_classes[0]}({top_probs[0]:.1f}%), "
                    confidence_text += f"{top_classes[1]}({top_probs[1]:.1f}%), "
                    confidence_text += f"{top_classes[2]}({top_probs[2]:.1f}%)"
                    
                    self.confidence_label.config(text=confidence_text)
                else:
                    self.confidence_label.config(text="")
            
            # Display prediction
            self.result_label.config(text=str(prediction))
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
