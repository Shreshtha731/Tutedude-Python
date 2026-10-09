import customtkinter as ctk
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
import tensorflow as tf
from landmarks import extract_keypoints, draw_styled_landmarks

# ---------------------------------------------------------
# Modern UI Theme Configuration
# ---------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HearMeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hear Me - Sign Language AI")
        self.geometry("1100x650")
        
        # Make the background slightly darker for a premium look
        self.configure(fg_color="#0F111A") 

        # --- AI & LOGIC VARIABLES ---
        self.sequence = []          
        self.current_sentence = []  
        self.predictions = []       
        self.threshold = 0.8        
        self.actions = np.array(['hello', 'no', 'i_love_you']) 
        
        try:
            self.model = tf.keras.models.load_model('action.h5')
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False

        self.mp_holistic = mp.solutions.holistic
        
        # --- UI LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR (Left)
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=20, fg_color="#1A1D27", border_width=1, border_color="#2A2D37")
        self.sidebar_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1) 

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Hear Me AI", font=ctk.CTkFont(size=32, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))

        # Fluid Loading Bar (Hidden by default)
        self.loading_bar = ctk.CTkProgressBar(self.sidebar_frame, width=200, height=8, corner_radius=5)
        self.loading_bar.set(0)

        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Turn Camera On", height=45, corner_radius=8, font=ctk.CTkFont(weight="bold"), command=self.animate_start_camera)
        self.start_btn.grid(row=2, column=0, padx=30, pady=(20, 10), sticky="ew")

        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Turn Camera Off", height=45, corner_radius=8, fg_color="#E53935", hover_color="#B71C1C", font=ctk.CTkFont(weight="bold"), command=self.stop_camera)
        self.stop_btn.grid(row=3, column=0, padx=30, pady=10, sticky="ew")
        
        self.clear_btn = ctk.CTkButton(self.sidebar_frame, text="Clear Translation", height=45, corner_radius=8, fg_color="#455A64", hover_color="#37474F", command=self.clear_text)
        self.clear_btn.grid(row=4, column=0, padx=30, pady=10, sticky="ew")

        # Status indicator (Used for pulsing animation)
        status_color = "#4CAF50" if self.model_loaded else "#F44336"
        status_text = "● Model Loaded" if self.model_loaded else "● No Model"
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text=status_text, text_color=status_color, font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.grid(row=5, column=0, padx=20, pady=20)

        # Output text box for translations
        self.output_box = ctk.CTkTextbox(self.sidebar_frame, height=180, corner_radius=12, fg_color="#101218", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00E676")
        self.output_box.grid(row=6, column=0, padx=20, pady=20, sticky="s")
        self.output_box.insert("0.0", " Waiting...")

        # 2. VIDEO FRAME (Right Side)
        self.video_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#1A1D27", border_width=1, border_color="#2A2D37")
        self.video_frame.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Camera Offline", text_color="#546E7A", font=ctk.CTkFont(size=24))
        self.video_label.pack(expand=True, fill="both", padx=15, pady=15)

        # Hardware variables
        self.cap = None
        self.camera_running = False

    # --- ANIMATION: Smooth Camera Boot ---
    def animate_start_camera(self):
        if not self.camera_running:
            self.start_btn.configure(state="disabled", text="Initializing hardware...")
            self.loading_bar.grid(row=1, column=0, pady=(0, 10))
            self.loading_bar.set(0)
            self.fill_loading_bar(0)

    def fill_loading_bar(self, value):
        if value < 1.0:
            value += 0.05
            self.loading_bar.set(value)
            self.after(25, self.fill_loading_bar, value) # Loop until full
        else:
            self.loading_bar.grid_forget() # Hide the bar
            self.start_btn.configure(state="normal", text="Turn Camera On")
            self.execute_camera_start()

    def execute_camera_start(self):
        self.cap = cv2.VideoCapture(0) 
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # Prevent WSL Timeout
        self.camera_running = True
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.video_label.configure(text="")
        
        self.pulse_live_indicator() # Start the pulsing red dot
        self.update_frame()

    # --- ANIMATION: Pulsing "LIVE" Dot ---
    def pulse_live_indicator(self):
        if self.camera_running:
            current_color = self.status_label.cget("text_color")
            # Toggle between bright red and very dark red
            new_color = "#FF1744" if current_color == "#5C0011" else "#5C0011"
            self.status_label.configure(text_color=new_color, text="🔴 LIVE - Tracking")
            self.after(700, self.pulse_live_indicator) 
        else:
            status_color = "#4CAF50" if self.model_loaded else "#F44336"
            status_text = "● Model Loaded" if self.model_loaded else "● No Model"
            self.status_label.configure(text_color=status_color, text=status_text)

    # --- ANIMATION: Fluid Typewriter Text ---
    def type_new_word(self, word, index=0):
        if index == 0:
            self.output_box.insert("end", " ") # Add a space before the new word
        if index < len(word):
            self.output_box.insert("end", word[index])
            # Delay slightly between each letter for fluid typing effect
            self.after(40, self.type_new_word, word, index + 1)

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'holistic'):
            self.holistic.close()
        self.video_label.configure(image="", text="Camera Offline")

    def clear_text(self):
        self.current_sentence = []
        self.output_box.delete("0.0", "end")
        self.output_box.insert("0.0", " Waiting...")

    # --- MAIN VIDEO LOOP ---
    def update_frame(self):
        if self.camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False 
                results = self.holistic.process(image)
                image.flags.writeable = True

                draw_styled_landmarks(image, results)
                
                if self.model_loaded:
                    keypoints = extract_keypoints(results)
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-30:] 

                    if len(self.sequence) == 30:
                        res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                        predicted_action = self.actions[np.argmax(res)]
                        self.predictions.append(np.argmax(res))

                        if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > self.threshold:
                                
                                # If it's a new word, clear placeholder and type it!
                                if len(self.current_sentence) == 0 or self.actions[np.argmax(res)] != self.current_sentence[-1]:
                                    if "Waiting..." in self.output_box.get("0.0", "end"):
                                        self.output_box.delete("0.0", "end") # Clear placeholder
                                    
                                    self.current_sentence.append(predicted_action)
                                    # Trigger the fluid typewriter animation
                                    self.type_new_word(predicted_action)

                        if len(self.current_sentence) > 5:
                            self.current_sentence = self.current_sentence[-5:]

                image = cv2.flip(image, 1) 
                frame_resized = cv2.resize(image, (800, 600))
                
                img_pil = Image.fromarray(frame_resized)
                img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(800, 600))

                self.video_label.configure(image=img_ctk)
                self.video_label.image = img_ctk 

            self.after(10, self.update_frame)

if __name__ == "__main__":
    app = HearMeApp()
    app.mainloop()