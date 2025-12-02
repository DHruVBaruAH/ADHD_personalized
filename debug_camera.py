import cv2
import numpy as np

def diagnose():
    print("🕵️ DIAGNOSING CAMERA BLACK SCREEN...")
    
    # Try different indices and backends
    configs = [
        (0, cv2.CAP_DSHOW, "Index 0 - DirectShow"),
        (0, cv2.CAP_MSMF,  "Index 0 - MediaFoundation"),
        (1, cv2.CAP_DSHOW, "Index 1 - DirectShow"),
        (1, cv2.CAP_MSMF,  "Index 1 - MediaFoundation"),
    ]

    for index, backend, name in configs:
        print(f"\n--- Testing: {name} ---")
        cap = cv2.VideoCapture(index, backend)
        
        if not cap.isOpened():
            print("❌ Failed to open.")
            continue

        # WARMUP: Cameras need time to adjust light (read 20 frames)
        print("   Warmup (reading 20 frames)...")
        for _ in range(20):
            cap.read()

        # Read actual frame
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Camera opened, but returned 'None' (No Frame).")
        else:
            # Check if image is truly black (sum of all pixels)
            total_brightness = np.sum(frame)
            if total_brightness == 0:
                print("⚠️  Frame received, but it is PURE BLACK (Values are all 0).")
                print("    -> Solution: Check Physical Privacy Shutter or F-Key.")
            else:
                print(f"✅ SUCCESS! Frame received. Brightness Score: {total_brightness}")
                print("   (Opening window for 5 seconds...)")
                cv2.imshow(f"Test: {name}", frame)
                cv2.waitKey(5000) 
                cv2.destroyAllWindows()
                cap.release()
                return # Stop if we found a working one

        cap.release()

    print("\n🛑 DIAGNOSIS COMPLETE: No working video feed found.")

if __name__ == "__main__":
    diagnose()