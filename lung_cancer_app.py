"""
Lung Cancer Detection GUI Application
Compatible with Python 3.9 - 3.12 (TensorFlow requirement)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import sys

# Fix matplotlib backend for Windows before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to import TensorFlow with helpful error message
try:
    import tensorflow as tf
except ModuleNotFoundError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "TensorFlow Not Found",
        "TensorFlow is not installed or not compatible with your Python version.\n\n"
        "REQUIREMENTS:\n"
        "- Python 3.9, 3.10, 3.11, or 3.12 (Python 3.13 is NOT supported)\n"
        "- Install TensorFlow: pip install tensorflow\n\n"
        "Please install a compatible Python version and try again."
    )
    sys.exit(1)

# Configuration
MODEL_PATH = "lung_cancer_model_resnet50.keras"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.1

class LungCancerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Detection System")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f2f5")
        self.root.minsize(800, 600)
        
        # Initialize model
        self.model = None
        self.load_model()
        
        # Setup UI
        self.setup_styles()
        self.create_widgets()
        
    def load_model(self):
        """Load the pre-trained model with error handling"""
        try:
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror(
                    "Model Not Found",
                    f"Model file not found:\n{MODEL_PATH}\n\n"
                    f"Please ensure the model file is in the same directory as this script:\n"
                    f"{os.path.dirname(os.path.abspath(__file__))}"
                )
                self.root.destroy()
                sys.exit(1)
            
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{str(e)}")
            self.root.destroy()
            sys.exit(1)
    
    def setup_styles(self):
        """Configure ttk styles"""
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Segoe UI", 11), padding=8)
        self.style.configure("TLabel", font=("Segoe UI", 11), background="#f0f2f5")
        self.style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), background="#2c3e50", foreground="white")
        self.style.configure("Result.TLabel", font=("Segoe UI", 14), background="#ffffff", padding=10)
        self.style.configure("Card.TFrame", background="#ffffff")
        
    def create_widgets(self):
        """Create all UI components"""
        # Header
        self.create_header()
        
        # Main Container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill="both", expand=True)
        
        # Left Panel (Image & Results)
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Image Display Area
        self.create_image_section(left_panel)
        
        # Results Section
        self.create_results_section(left_panel)
        
        # Graph Section
        self.create_graph_section(left_panel)
        
        # Right Panel (Controls)
        right_panel = ttk.Frame(main_container, width=200)
        right_panel.pack(side="right", fill="y", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        self.create_control_panel(right_panel)
        
        # Footer
        self.create_footer()
        
    def create_header(self):
        """Create application header"""
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title = tk.Label(
            header, 
            text="🫁 Lung Cancer Detection System", 
            font=("Segoe UI", 22, "bold"),
            bg="#2c3e50", 
            fg="white"
        )
        title.pack(pady=10)
        
        subtitle = tk.Label(
            header,
            text="AI-Powered Medical Image Analysis",
            font=("Segoe UI", 10),
            bg="#2c3e50",
            fg="#bdc3c7"
        )
        subtitle.pack()
        
    def create_image_section(self, parent):
        """Create image upload and display section"""
        # Image Card
        image_card = tk.Frame(parent, bg="white", bd=1, relief="solid", highlightbackground="#ddd")
        image_card.pack(fill="x", pady=(0, 15))
        
        # Image Label
        self.image_label = tk.Label(
            image_card,
            text="📤 Drop or Upload an Image\n\nSupported formats: PNG, JPG, JPEG\nRecommended size: 224x224 pixels",
            font=("Segoe UI", 12),
            bg="#f8f9fa",
            fg="#7f8c8d",
            width=50,
            height=15,
            relief="flat"
        )
        self.image_label.pack(padx=2, pady=2, fill="both", expand=True)
        
    def create_results_section(self, parent):
        """Create prediction results display"""
        # Results Card
        results_card = tk.Frame(parent, bg="white", bd=1, relief="solid", highlightbackground="#ddd")
        results_card.pack(fill="x", pady=(0, 15))
        
        # Results Header
        results_header = tk.Label(
            results_card,
            text="📊 Analysis Results",
            font=("Segoe UI", 12, "bold"),
            bg="#3498db",
            fg="white",
            pady=5
        )
        results_header.pack(fill="x")
        
        # Result Label
        self.result_label = tk.Label(
            results_card,
            text="No analysis performed yet.\nUpload an image to begin detection.",
            font=("Segoe UI", 13),
            bg="white",
            fg="#2c3e50",
            wraplength=500,
            justify="center",
            pady=20
        )
        self.result_label.pack(fill="x")
        
        # Confidence Label
        self.confidence_label = tk.Label(
            results_card,
            text="",
            font=("Segoe UI", 11),
            bg="white",
            fg="#7f8c8d"
        )
        self.confidence_label.pack(fill="x", pady=(0, 10))
        
    def create_graph_section(self, parent):
        """Create probability graph section"""
        # Graph Card
        self.graph_card = tk.Frame(parent, bg="white", bd=1, relief="solid", highlightbackground="#ddd")
        self.graph_card.pack(fill="x")
        
        # Graph Header
        graph_header = tk.Label(
            self.graph_card,
            text="📈 Probability Distribution",
            font=("Segoe UI", 12, "bold"),
            bg="#27ae60",
            fg="white",
            pady=5
        )
        graph_header.pack(fill="x")
        
        # Graph Frame
        self.graph_frame = tk.Frame(self.graph_card, bg="white", height=250)
        self.graph_frame.pack(fill="x", padx=10, pady=10)
        self.graph_frame.pack_propagate(False)
        
    def create_control_panel(self, parent):
        """Create control buttons panel"""
        # Control Card
        control_card = tk.Frame(parent, bg="white", bd=1, relief="solid", highlightbackground="#ddd")
        control_card.pack(fill="both", expand=True)
        
        # Control Header
        control_header = tk.Label(
            control_card,
            text="🎛️ Controls",
            font=("Segoe UI", 12, "bold"),
            bg="#9b59b6",
            fg="white",
            pady=8
        )
        control_header.pack(fill="x")
        
        # Button Container
        btn_container = tk.Frame(control_card, bg="white", padx=15, pady=15)
        btn_container.pack(fill="both", expand=True)
        
        # Upload Button
        self.upload_btn = tk.Button(
            btn_container,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Segoe UI", 11, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            pady=10
        )
        self.upload_btn.pack(fill="x", pady=(0, 10))
        
        # Try Another Button
        self.try_again_btn = tk.Button(
            btn_container,
            text="🔄 Try Another",
            command=self.reset_analysis,
            font=("Segoe UI", 11),
            bg="#95a5a6",
            fg="white",
            activebackground="#7f8c8d",
            state="disabled",
            relief="flat",
            cursor="hand2",
            pady=8
        )
        self.try_again_btn.pack(fill="x", pady=(0, 10))
        
        # Separator
        tk.Frame(btn_container, bg="#ecf0f1", height=2).pack(fill="x", pady=10)
        
        # Save Report Button
        self.save_btn = tk.Button(
            btn_container,
            text="💾 Save Report",
            command=self.save_report,
            font=("Segoe UI", 11),
            bg="#27ae60",
            fg="white",
            activebackground="#229954",
            state="disabled",
            relief="flat",
            cursor="hand2",
            pady=8
        )
        self.save_btn.pack(fill="x", pady=(0, 10))
        
        # Separator
        tk.Frame(btn_container, bg="#ecf0f1", height=2).pack(fill="x", pady=10)
        
        # About Button
        about_btn = tk.Button(
            btn_container,
            text="ℹ️ About",
            command=self.show_about,
            font=("Segoe UI", 11),
            bg="#f39c12",
            fg="white",
            activebackground="#d68910",
            relief="flat",
            cursor="hand2",
            pady=8
        )
        about_btn.pack(fill="x", pady=(0, 10))
        
        # Exit Button
        exit_btn = tk.Button(
            btn_container,
            text="❌ Exit",
            command=self.exit_app,
            font=("Segoe UI", 11),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            relief="flat",
            cursor="hand2",
            pady=8
        )
        exit_btn.pack(fill="x")
        
    def create_footer(self):
        """Create application footer"""
        footer = tk.Frame(self.root, bg="#2c3e50", height=30)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        
        footer_text = tk.Label(
            footer,
            text="Powered by TensorFlow & ResNet50 | Developed for Medical Research",
            font=("Segoe UI", 9),
            bg="#2c3e50",
            fg="#95a5a6"
        )
        footer_text.pack(pady=5)
        
    def predict_image(self, image_path):
        """
        Process image and make prediction using the loaded model
        Returns: (cancer_prob, healthy_prob, health_report, status)
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize to model input size
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img)
            
            # Ensure correct shape (224, 224, 3)
            if img_array.shape != (224, 224, 3):
                raise ValueError(f"Invalid image shape: {img_array.shape}. Expected (224, 224, 3)")
            
            # Normalize and add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            cancer_probability = float(prediction[0][0])
            healthy_probability = 1.0 - cancer_probability
            
            # Generate health report
            if cancer_probability > THRESHOLD:
                status = "positive"
                health_report = f"⚠️  POSITIVE: Potential malignancy detected"
                details = f"Cancer probability: {cancer_probability * 100:.2f}%"
            else:
                status = "negative"
                health_report = f"✅ NEGATIVE: No significant abnormalities detected"
                details = f"Healthy tissue probability: {healthy_probability * 100:.2f}%"
            
            return cancer_probability, healthy_probability, health_report, status, details
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error analyzing image:\n{str(e)}")
            return None, None, None, None, None
            
    def upload_image(self):
        """Handle image upload and analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        # Update UI to show processing
        self.image_label.config(text="⏳ Processing image...", fg="#3498db")
        self.root.update()
        
        # Perform prediction
        result = self.predict_image(file_path)
        
        if result[0] is None:
            self.image_label.config(
                text="❌ Error processing image\nPlease try another file",
                fg="#e74c3c"
            )
            return
            
        cancer_prob, healthy_prob, report, status, details = result
        
        # Update results display
        self.display_results(file_path, cancer_prob, healthy_prob, report, status, details)
        
        # Enable buttons
        self.try_again_btn.config(state="normal", bg="#7f8c8d")
        self.save_btn.config(state="normal")
        
    def display_results(self, image_path, cancer_prob, healthy_prob, report, status, details):
        """Update UI with analysis results"""
        # Display image
        img = Image.open(image_path)
        img.thumbnail((400, 400))
        img_display = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img_display, text="", bg="white")
        self.image_label.image = img_display  # Keep reference
        
        # Update result text with color coding
        if status == "positive":
            bg_color = "#ffebee"
            fg_color = "#c0392b"
        else:
            bg_color = "#e8f5e9"
            fg_color = "#27ae60"
            
        self.result_label.config(
            text=f"{report}\n\n{details}",
            bg=bg_color,
            fg=fg_color,
            font=("Segoe UI", 13, "bold")
        )
        
        self.confidence_label.config(
            text=f"Model confidence: {max(cancer_prob, healthy_prob) * 100:.2f}%"
        )
        
        # Update graph
        self.plot_graph(healthy_prob, cancer_prob, status)
        
    def plot_graph(self, healthy_prob, cancer_prob, status):
        """Create probability bar chart"""
        # Clear previous graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        
        # Data
        categories = ['Healthy\nTissue', 'Potential\nMalignancy']
        probabilities = [healthy_prob * 100, cancer_prob * 100]
        
        # Colors based on prediction
        if status == "positive":
            colors = ['#95a5a6', '#e74c3c']
        else:
            colors = ['#27ae60', '#95a5a6']
            
        # Create bars
        bars = ax.bar(categories, probabilities, color=colors, width=0.6, edgecolor='white', linewidth=2)
        
        # Customize
        ax.set_ylabel('Probability (%)', fontsize=11, fontweight='bold')
        ax.set_title('Prediction Confidence', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 2,
                f'{prob:.1f}%',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )
            
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()
        
        # Close figure to prevent memory issues
        plt.close(fig)
        
    def reset_analysis(self):
        """Reset the application for new analysis"""
        # Clear image
        self.image_label.config(
            image="",
            text="📤 Drop or Upload an Image\n\nSupported formats: PNG, JPG, JPEG\nRecommended size: 224x224 pixels",
            bg="#f8f9fa",
            fg="#7f8c8d"
        )
        self.image_label.image = None
        
        # Reset results
        self.result_label.config(
            text="No analysis performed yet.\nUpload an image to begin detection.",
            bg="white",
            fg="#2c3e50",
            font=("Segoe UI", 13)
        )
        self.confidence_label.config(text="")
        
        # Clear graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # Disable buttons
        self.try_again_btn.config(state="disabled", bg="#95a5a6")
        self.save_btn.config(state="disabled")
        
    def save_report(self):
        """Save analysis report to file"""
        report_text = self.result_label.cget("text")
        confidence_text = self.confidence_label.cget("text")
        
        if "No analysis performed" in report_text:
            messagebox.showwarning("No Data", "Please perform an analysis before saving.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ],
            title="Save Analysis Report"
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write("LUNG CANCER DETECTION REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Result: {report_text}\n")
                    f.write(f"{confidence_text}\n\n")
                    f.write("-" * 50 + "\n")
                    f.write("Generated by Lung Cancer Detection System\n")
                    f.write("This report is for research purposes only.\n")
                    f.write("Please consult a medical professional for diagnosis.\n")
                    
                messagebox.showinfo("Success", f"Report saved successfully!\n\nLocation: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report:\n{str(e)}")
                
    def show_about(self):
        """Display about dialog"""
        about_text = """Lung Cancer Detection System v1.0

🔬 Technology Stack:
   • TensorFlow 2.x with ResNet50 Architecture
   • Python Tkinter GUI
   • PIL/Pillow Image Processing
   • Matplotlib Visualization

⚠️  Important Disclaimer:
This application is designed for research and educational 
purposes only. It is NOT a substitute for professional 
medical diagnosis. Always consult qualified healthcare 
providers for medical advice and diagnosis.

📧 For support or inquiries, contact the development team.
"""
        messagebox.showinfo("About", about_text)
        
    def exit_app(self):
        """Clean exit with confirmation"""
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            self.root.destroy()
            sys.exit(0)


def main():
    """Main application entry point"""
    # Check Python version compatibility
    version = sys.version_info
    if version.major != 3 or version.minor not in [9, 10, 11, 12]:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Python Version Error",
            f"Python {version.major}.{version.minor} is not compatible.\n\n"
            f"REQUIRED: Python 3.9, 3.10, 3.11, or 3.12\n"
            f"YOUR VERSION: Python {version.major}.{version.minor}.{version.micro}\n\n"
            f"TensorFlow does not support Python 3.13 yet.\n"
            f"Please install a compatible Python version."
        )
        sys.exit(1)
    
    # Create main window
    root = tk.Tk()
    
    # Set DPI awareness for Windows (crisp text)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    # Start application
    app = LungCancerDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()