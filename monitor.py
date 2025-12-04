import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- CONFIGURATION ---
STATUS_FILE = "monitor_status.txt"

class HyperactivityMonitor:
    def __init__(self, threshold=0.5, alert_cooldown=10):
        print("🧠 Initializing AI Engine (MediaPipe)...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.prev_landmarks = None
        self.movement_history = []
        self.last_alert_time = 0
        self.threshold = threshold
        self.alert_cooldown = alert_cooldown

        # Reset status file
        with open(STATUS_FILE, "w") as f:
            f.write("Normal")

    def calculate_movement(self, current_landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return 0
        
        key_indices = [0, 11, 12, 15, 16]
        displacement = 0
        for idx in key_indices:
            curr = np.array([current_landmarks[idx].x, current_landmarks[idx].y])
            prev = np.array([self.prev_landmarks[idx].x, self.prev_landmarks[idx].y])
            displacement += np.linalg.norm(curr - prev)
            
        self.prev_landmarks = current_landmarks
        return displacement / len(key_indices)

    def trigger_alert(self):
        print("🚨 WRITING ALERT TO FILE...") 
        with open(STATUS_FILE, "w") as f:
            f.write("ALERT: High Hyperactivity Detected!")

    def clear_alert(self):
        with open(STATUS_FILE, "w") as f:
            f.write("Normal")

    def process_frame(self, frame):
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        try:
            results = self.pose.process(image)
        except Exception as e:
            print(f"MediaPipe Error: {e}")
            return frame

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        status = "Normal"
        color = (0, 255, 0)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            movement = self.calculate_movement(results.pose_landmarks.landmark)
            self.movement_history.append(movement)
            if len(self.movement_history) > 50: 
                self.movement_history.pop(0)
            
            avg_movement = np.mean(self.movement_history)

            if avg_movement > self.threshold:
                status = "High Activity"
                color = (0, 0, 255)
                if time.time() - self.last_alert_time > self.alert_cooldown:
                    self.trigger_alert()
                    self.last_alert_time = time.time()
            else:
                if time.time() - self.last_alert_time > 5:
                    self.clear_alert()

            cv2.putText(image, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Energy: {avg_movement:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return image

def find_working_camera():
    """Scans for a working camera exactly like debug_camera.py did"""
    print("\n🔍 AUTO-DETECTING CAMERA...")
    
    # Priority list of configurations to try
    configs = [
        (0, cv2.CAP_MSMF,  "Index 0 (MediaFoundation)"),
        (0, cv2.CAP_DSHOW, "Index 0 (DirectShow)"),
        (1, cv2.CAP_MSMF,  "Index 1 (MediaFoundation)"),
        (1, cv2.CAP_DSHOW, "Index 1 (DirectShow)"),
    ]

    for index, backend, name in configs:
        print(f"   Testing: {name}...", end=" ")
        
        # --- FIX IS HERE: Use 'backend', not hardcoded 'DSHOW' ---
        cap = cv2.VideoCapture(index, backend)

        if not cap.isOpened():
            print("Failed (Closed).")
            continue

        # Force standard resolution (Critical for MediaPipe)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warmup
        for _ in range(10): cap.read()
        
        ret, frame = cap.read()
        if ret and np.sum(frame) > 0: # Check if frame is not black
            print("SUCCESS! ✅")
            return cap
        else:
            print("Failed (Black Screen or No Data).")
            cap.release()
            
    return None

if __name__ == "__main__":
    # 1. Find the camera
    cap = find_working_camera()

    if cap is None:
        print("\n❌ CRITICAL ERROR: No working camera found.")
        print("   Please ensure Zoom/Teams are closed and permissions are on.")
        input("   Press Enter to exit...")
        exit()

    print("✅ Camera locked. Starting Hyperactivity Monitor...")
    print("   (Press 'q' to quit)")

    # 2. Start the Logic
    monitor = HyperactivityMonitor(threshold=0.01) # Lower threshold for easier testing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream ended.")
            break
        
        try:
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process
            processed_frame = monitor.process_frame(frame)
            cv2.imshow('ADHD Hyperactivity Guard', processed_frame)
        except Exception as e:
            print(f"Error loop: {e}")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()