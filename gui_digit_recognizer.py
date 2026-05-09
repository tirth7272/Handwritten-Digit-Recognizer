import tkinter as tk
from tkinter import *
import win32gui

import numpy as np

from PIL import ImageGrab, Image, ImageOps

from tensorflow.keras.models import load_model


# =========================
# LOAD TRAINED MODEL
# =========================

model = load_model('mnist_trained.h5')


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(img):

    # Convert to grayscale
    img = img.convert('L')

    # Invert colors
    img = ImageOps.invert(img)

    # Convert to numpy array
    img = np.array(img)

    # Apply threshold
    img = np.where(img > 50, 255, 0).astype(np.uint8)

    # Find non-empty pixels
    coords = np.argwhere(img)

    if coords.size == 0:
        return np.zeros((1, 28, 28, 1))

    # Bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop digit
    img = img[y0:y1, x0:x1]

    # Convert back to PIL
    img = Image.fromarray(img)

    # Maintain aspect ratio
    width, height = img.size

    if width > height:

        new_width = 20
        new_height = int((20 / width) * height)

    else:

        new_height = 20
        new_width = int((20 / height) * width)

    img = img.resize((new_width, new_height))

    # Create 28x28 black image
    new_img = Image.new('L', (28, 28), 0)

    # Center image
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2

    new_img.paste(img, (paste_x, paste_y))

    # Normalize
    img = np.array(new_img).astype('float32') / 255.0

    # Reshape
    img = img.reshape(1, 28, 28, 1)

    return img


# =========================
# PREDICTION FUNCTION
# =========================

def predict_digit(img):

    processed_img = preprocess_image(img)

    prediction = model.predict(processed_img, verbose=0)[0]

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    return digit, confidence


# =========================
# MAIN APPLICATION CLASS
# =========================

class App(tk.Tk):

    def __init__(self):

        super().__init__()

        # =========================
        # WINDOW SETTINGS
        # =========================

        self.title("Handwritten Digit Recognizer")

        self.geometry("700x450")

        self.configure(bg="#f0f0f0")

        self.resizable(False, False)

        # =========================
        # LEFT FRAME
        # =========================

        left_frame = tk.Frame(self, bg="#f0f0f0")

        left_frame.pack(side="left", padx=20, pady=20)

        # =========================
        # DRAWING CANVAS
        # =========================

        self.canvas = tk.Canvas(
            left_frame,
            width=300,
            height=300,
            bg='white',
            cursor='cross',
            bd=2,
            relief='solid'
        )

        self.canvas.pack()

        # =========================
        # BUTTON FRAME
        # =========================

        button_frame = tk.Frame(left_frame, bg="#f0f0f0")

        button_frame.pack(pady=15)

        # =========================
        # CLEAR BUTTON
        # =========================

        self.clear_btn = tk.Button(
            button_frame,
            text='Clear',
            command=self.clear_all,
            width=12,
            font=('Arial', 12)
        )

        self.clear_btn.grid(row=0, column=0, padx=10)

        # =========================
        # RECOGNIZE BUTTON
        # =========================

        self.classify_btn = tk.Button(
            button_frame,
            text='Recognize',
            command=self.classify_handwriting,
            width=12,
            font=('Arial', 12)
        )

        self.classify_btn.grid(row=0, column=1, padx=10)

        # =========================
        # RIGHT FRAME
        # =========================

        right_frame = tk.Frame(self, bg="#f0f0f0")

        right_frame.pack(side="right", padx=30)

        # =========================
        # PREDICTION LABEL
        # =========================

        self.label = tk.Label(
            right_frame,
            text='Draw a Digit',
            bg="#f0f0f0",
            font=('Helvetica', 24, 'bold'),
            justify='center'
        )

        self.label.pack()

        # =========================
        # BIND DRAWING EVENT
        # =========================

        self.canvas.bind('<B1-Motion>', self.draw_lines)

    # =========================
    # CLEAR CANVAS
    # =========================

    def clear_all(self):

        self.canvas.delete('all')

        self.label.configure(text='Draw a Digit')

    # =========================
    # DRAW ON CANVAS
    # =========================

    def draw_lines(self, event):

        x1 = event.x - 7
        y1 = event.y - 7

        x2 = event.x + 7
        y2 = event.y + 7

        self.canvas.create_oval(
            x1,
            y1,
            x2,
            y2,
            fill='black',
            outline='black'
        )

    # =========================
    # CLASSIFY HANDWRITING
    # =========================

    def classify_handwriting(self):

        HWND = self.canvas.winfo_id()

        rect = win32gui.GetWindowRect(HWND)

        x, y, w, h = rect

        rect = (x + 4, y + 4, w - 4, h - 4)

        img = ImageGrab.grab(rect)

        digit, confidence = predict_digit(img)

        self.label.configure(
            text=f'Prediction: {digit}\nConfidence: {int(confidence * 100)}%'
        )


# =========================
# RUN APPLICATION
# =========================

app = App()

app.mainloop()