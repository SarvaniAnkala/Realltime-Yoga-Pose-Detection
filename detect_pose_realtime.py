import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
import math

# Load model, label encoder, and feature names
model = joblib.load("yoga_pose_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def give_feedback(pose_label, landmarks):
    feedback = []

    def get_coords(name):
        lm = mp_pose.PoseLandmark[name]
        return [landmarks[lm].x, landmarks[lm].y]

    if pose_label == "plank":
        left_shoulder = get_coords('LEFT_SHOULDER')
        left_hip = get_coords('LEFT_HIP')
        left_ankle = get_coords('LEFT_ANKLE')
        angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        if angle < 160 or angle > 180:
            feedback.append("Keep your body in a straight line (plank)")

    elif pose_label == "dogdown":
        left_hip = get_coords('LEFT_HIP')
        left_shoulder = get_coords('LEFT_SHOULDER')
        left_wrist = get_coords('LEFT_WRIST')
        angle = calculate_angle(left_hip, left_shoulder, left_wrist)
        if angle < 150:
            feedback.append("Push your chest closer to your thighs")

    elif pose_label == "goddess":
        left_knee = get_coords('LEFT_KNEE')
        left_hip = get_coords('LEFT_HIP')
        left_ankle = get_coords('LEFT_ANKLE')
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if knee_angle < 80 or knee_angle > 100:
            feedback.append("Bend knees to 90°")

    elif pose_label == "warrior2":
        left_knee = get_coords('LEFT_KNEE')
        left_hip = get_coords('LEFT_HIP')
        left_ankle = get_coords('LEFT_ANKLE')
        front_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if front_knee_angle < 80 or front_knee_angle > 100:
            feedback.append("Bend front knee to 90°")
        left_shoulder = get_coords('LEFT_SHOULDER')
        left_elbow = get_coords('LEFT_ELBOW')
        left_wrist = get_coords('LEFT_WRIST')
        arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if arm_angle < 160:
            feedback.append("Straighten your arms")

    elif pose_label == "tree":
        left_knee = get_coords('LEFT_KNEE')
        left_hip = get_coords('LEFT_HIP')
        left_ankle = get_coords('LEFT_ANKLE')
        standing_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if standing_leg_angle < 170:
            feedback.append("Keep your standing leg straight")

    return feedback

# Start webcam
cap = cv2.VideoCapture(0)

# Webcam loop
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Webcam not detected.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])

            if len(keypoints) == 99:
                input_df = pd.DataFrame([keypoints], columns=feature_names)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0]
                    confidence = max(proba)
                    prediction = np.argmax(proba)

                    if confidence > 0.4:
                        predicted_label = label_encoder.inverse_transform([prediction])[0]
                        cv2.putText(image, f'Pose: {predicted_label}', (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Optional: Show confidence
                        cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                        feedback_list = give_feedback(predicted_label.lower(), landmarks)
                        for idx, feedback in enumerate(feedback_list):
                            cv2.putText(image, feedback, (10, 110 + 30 * idx),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(image, "No valid pose detected", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    prediction = model.predict(input_df)[0]
                    predicted_label = label_encoder.inverse_transform([prediction])[0]
                    cv2.putText(image, f'Pose: {predicted_label}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Yoga Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
