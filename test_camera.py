import cv2

def test_camera():
    print("🔍 Scanning for cameras...")
    
    # Try indices 0, 1, 2 (in case you have virtual cameras like OBS)
    for index in range(3):
        print(f"\nTesting Camera Index [{index}]...")
        
        # Try with DirectShow (Best for Windows)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"❌ Camera [{index}] failed to open.")
        else:
            ret, frame = cap.read()
            if ret:
                print(f"✅ SUCCESS! Camera [{index}] is working.")
                print("   (Press 'q' to quit the camera window)")
                
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    cv2.imshow(f'Test Camera {index}', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return  # Stop after finding the working camera
            else:
                print(f"⚠️ Camera [{index}] opened but returned no image.")
    
    print("\n❌ NO WORKING CAMERAS FOUND.")
    print("👉 Check: Is another app (Zoom/Teams) using the camera?")
    print("👉 Check: Windows Settings > Privacy > Camera > 'Allow desktop apps to access your camera'")

if __name__ == "__main__":
    test_camera()