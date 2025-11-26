"""
Handwritten Character Recognition GUI
Using Tkinter Canvas for drawing.
Decoupled from model logic.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps

class HandwritingRecognitionApp:
    def __init__(self, root, prediction_callback=None):
        self.root = root
        self.root.title("Handwritten Character Recognition")
        self.root.geometry("800x850")
        self.root.resizable(True, True)
        
        self.prediction_callback = prediction_callback
        
        # Drawing variables
        self.canvas_size = 400
        self.brush_size = 20
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets with scrollbar"""
        # Main container for canvas and scrollbar
        container = tk.Frame(self.root, bg="#f0f0f0")
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas
        self.main_canvas = tk.Canvas(container, bg="#f0f0f0")
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.main_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        self.main_canvas.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        
        # Create frame inside canvas
        main_frame = tk.Frame(self.main_canvas, bg="#f0f0f0")
        
        # Add frame to canvas window
        self.main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Handwritten Character Recognition",
            font=("Arial", 24, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=(20, 20), padx=20)
        
        # Canvas frame
        canvas_frame = tk.Frame(main_frame, bg="#f0f0f0")
        canvas_frame.pack(padx=20)
        
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
        instructions.pack(pady=10, padx=20)
        
        # Brush size control
        brush_frame = tk.Frame(main_frame, bg="#f0f0f0")
        brush_frame.pack(pady=10, padx=20)
        
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
        button_frame.pack(pady=20, padx=20)
        
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
        result_frame.pack(pady=20, padx=20)
        
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
        self.confidence_label.pack(padx=20, pady=(0, 20))
        
        # Bind mouse wheel to scroll
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # For Linux
        self.main_canvas.bind_all("<Button-4>", lambda e: self.main_canvas.yview_scroll(-1, "units"))
        self.main_canvas.bind_all("<Button-5>", lambda e: self.main_canvas.yview_scroll(1, "units"))
    
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
        
        return image_array
    
    def update_result(self, prediction_text, confidence_text):
        """Update the result labels"""
        self.result_label.config(text=prediction_text)
        self.confidence_label.config(text=confidence_text)

    def predict(self):
        """Prepare image and call callback"""
        # Preprocess image
        processed_image = self.preprocess_image()
        
        if processed_image is None:
            messagebox.showwarning("Warning", "Please draw something first!")
            return
            
        # print("TEST", processed_image)
        
        if self.prediction_callback:
            self.prediction_callback(processed_image)
        else:
            messagebox.showerror("Error", "No prediction callback registered!")

if __name__ == "__main__":
    # Test mode (no prediction)
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()
