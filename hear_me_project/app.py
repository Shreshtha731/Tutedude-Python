import customtkinter as ctk
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import your core pipeline modules
from landmarks import LandmarkExtractor
from recognition_interface import SignRecognizer

# UI Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HearMeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Configuration
        self.title("The Hear Me Project - Live Dashboard (WSL)")
        self.geometry("1280x720")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize Core Pipeline Modules
        self.extractor = LandmarkExtractor()
        self.recognizer = SignRecognizer()

        # Camera Initialization - Linux/WSL Standard
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Build the Dashboard Layout
        self._create_layout()

        # Start the video and inference loop
        self.update_frame()

    def _create_layout(self):
        """Architects the dual-column desktop dashboard."""
        self.grid_columnconfigure(0, weight=3) # Live Viewport
        self.grid_columnconfigure(1, weight=1) # Telemetry Dashboard
        self.grid_rowconfigure(0, weight=1)

        # --- Column 1: Live Viewport ---
        self.viewport_frame = ctk.CTkFrame(self, corner_radius=10)
        self.viewport_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = tk.Label(self.viewport_frame, bg="#2b2b2b")
        self.video_label.pack(expand=True, fill="both")

        # --- Column 2: Telemetry Dashboard ---
        self.dashboard_frame = ctk.CTkFrame(self, corner_radius=10)
        self.dashboard_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.dashboard_frame, text="Translation Telemetry", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=20)

        self.confidence_label = ctk.CTkLabel(self.dashboard_frame, text="Confidence Level: 0%", font=ctk.CTkFont(size=14))
        self.confidence_label.pack(pady=(20, 5))
        self.confidence_gauge = ctk.CTkProgressBar(self.dashboard_frame, width=250)
        self.confidence_gauge.set(0)
        self.confidence_gauge.pack(pady=10)

        self.state_label = ctk.CTkLabel(self.dashboard_frame, text="Active State: Waiting...", font=ctk.CTkFont(size=16))
        self.state_label.pack(pady=20)

        self.subtitle_ribbon = ctk.CTkTextbox(self.dashboard_frame, width=280, height=200, font=ctk.CTkFont(size=18))
        self.subtitle_ribbon.pack(pady=20)
        self.subtitle_ribbon.insert("0.0", "System Initialized.\nWaiting for gestures...")
        self.subtitle_ribbon.configure(state="disabled")

        self.exit_btn = ctk.CTkButton(self.dashboard_frame, text="Shutdown System", fg_color="#c0392b", hover_color="#e74c3c", command=self.on_closing)
        self.exit_btn.pack(side="bottom", pady=20)

    def update_frame(self):
        """Main execution loop tracking landmarks, running inference, and upscaling UI."""
        success, frame = self.cap.read()
        if success:
            # 1. Kinematic Extraction
            raw_vector = self.extractor.extract_keypoints(frame)
            
            # --- Vector Sanitization & Type Casting ---
            sanitized_vector = []
            if isinstance(raw_vector, (list, tuple)):
                for item in raw_vector:
                    arr = np.array(item).flatten()
                    if len(arr) == 0:
                        sanitized_vector.extend(np.zeros(63))
                    else:
                        sanitized_vector.extend(arr)
            else:
                sanitized_vector = np.array(raw_vector).flatten()
                
            # Force standard float32 array for TensorFlow compatibility
            sanitized_vector = np.array(sanitized_vector, dtype=np.float32)
            
            if sanitized_vector.shape[0] == 0:
                sanitized_vector = np.zeros(126, dtype=np.float32)
            # ------------------------------------------
            
            # 2. Neural Inference & Telemetry Update
            try:
                result = self.recognizer.predict(sanitized_vector)
                if result:
                    label, conf = result if isinstance(result, tuple) else (result, 1.0)
                    
                    # Prevent 'NoneType' crash if model output is empty/low-confidence
                    if label:
                        self.confidence_label.configure(text=f"Confidence Level: {int(conf*100)}%")
                        self.confidence_gauge.set(float(conf))
                        self.state_label.configure(text=f"Active State: {label.upper()}")
                        
                        self.subtitle_ribbon.configure(state="normal")
                        self.subtitle_ribbon.delete("1.0", tk.END)
                        self.subtitle_ribbon.insert(tk.END, f">> {label.upper()} detected.\nConfidence: {int(conf*100)}%")
                        self.subtitle_ribbon.configure(state="disabled")
            except Exception as e:
                print(f"[DEBUG] Inference blocked: {e}")

            # 3. UI Upscaling
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            target_width = self.viewport_frame.winfo_width()
            target_height = self.viewport_frame.winfo_height()
            
            if target_width > 10 and target_height > 10:
                img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(33, self.update_frame)

    def on_closing(self):
        """Safe shutdown sequence."""
        print("[i] Shutting down The Hear Me Project...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = HearMeApp()
    app.mainloop()