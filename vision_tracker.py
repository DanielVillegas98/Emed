import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

def trigger_voice_triage():
    print("\n" + "="*40)
    print("🚨 FALL DETECTED! INTERFACING WITH VOICE TRIAGE... 🚨")
    print("="*40 + "\n")
    from voice_triage import run_voice_triage_loop
    run_voice_triage_loop()

def main():
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Pose Model...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False)
    
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0) 
    
    fall_triggered = False
    consecutive_fall_frames = 0
    FALL_FRAME_THRESHOLD = 3 # Lowered slightly for faster live response

    # 0.65 means 65% down from the top of the screen. Adjust this higher or lower if needed!
    DEMO_THRESHOLD_Y = 0.65 

    print("NeuroEdge Vision System Active. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw the visible threshold line on screen for the judges
        line_y_pixels = int(DEMO_THRESHOLD_Y * h)
        cv2.line(frame, (0, line_y_pixels), (w, line_y_pixels), (0, 0, 255), 2)
        cv2.putText(frame, "FALL SAFETY LIMIT", (10, line_y_pixels - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(mp_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            
            try:
                nose_y = landmarks[0].y
                left_shoulder_y = landmarks[11].y
                right_shoulder_y = landmarks[12].y
                
                # Draw tracking dots (blue for nose, green for shoulders)
                cv2.circle(frame, (int(landmarks[0].x * w), int(nose_y * h)), 6, (255, 0, 0), -1)
                cv2.circle(frame, (int(landmarks[11].x * w), int(left_shoulder_y * h)), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(landmarks[12].x * w), int(right_shoulder_y * h)), 5, (0, 255, 0), -1)
                
                # FOOLPROOF DEMO TRIGGER: If the blue nose dot drops below the red line
                if nose_y > DEMO_THRESHOLD_Y: 
                    consecutive_fall_frames += 1
                else:
                    consecutive_fall_frames = 0
                
                if consecutive_fall_frames >= FALL_FRAME_THRESHOLD and not fall_triggered:
                    cv2.putText(frame, "FALL DETECTED!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    fall_triggered = True
                    trigger_voice_triage()
                    
            except IndexError:
                pass

        if fall_triggered:
            cv2.putText(frame, "STATUS: EMERGENCY TRIAGE ACTIVE", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: AMBIENT MONITORING", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('NeuroEdge Hub - Edge AI Feed', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('r'):
            fall_triggered = False
            consecutive_fall_frames = 0
            print("System Reset. Re-monitoring...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()