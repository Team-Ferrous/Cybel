import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from generator import ImageGenerator
import threading
import os

class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uncensored Image Generator")
        self.generator = ImageGenerator()
        self.image_label = None
        self.create_widgets()

    def create_widgets(self):
        # Prompt entry
        tk.Label(self.root, text="Enter prompt:").pack(pady=5)
        self.prompt_entry = tk.Entry(self.root, width=50)
        self.prompt_entry.pack(pady=5)

        # Generate button
        self.generate_button = tk.Button(self.root, text="Generate Image", command=self.generate_image_thread)
        self.generate_button.pack(pady=10)

        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)

    def generate_image_thread(self):
        threading.Thread(target=self.generate_image).start()

    def generate_image(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt.")
            return

        self.status_label.config(text="Generating...")
        self.generate_button.config(state=tk.DISABLED)

        output_path = "generated_image.png"
        result = self.generator.generate_image(prompt, output_path)

        if result and os.path.exists(result):
            self.display_image(result)
            self.status_label.config(text="Image generated successfully!")
        else:
            self.status_label.config(text="Error generating image.")
            messagebox.showerror("Error", "Failed to generate image.")

        self.generate_button.config(state=tk.NORMAL)

    def display_image(self, path):
        try:
            image = Image.open(path)
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()