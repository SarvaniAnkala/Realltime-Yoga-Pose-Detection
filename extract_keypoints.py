import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

DATA_DIR = './DATASET'  # Change this!
SETS = ['TRAIN', 'TEST']
all_data = []

for dataset_type in SETS:
    print(f"\nProcessing {dataset_type} data...\n")
    dataset_path = os.path.join(DATA_DIR, dataset_type)
    
    for pose_label in os.listdir(dataset_path):
        pose_dir = os.path.join(dataset_path, pose_label)
        
        for img_file in tqdm(os.listdir(pose_dir), desc=f"{pose_label}"):
            img_path = os.path.join(pose_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                keypoints = []
                for lm in results.pose_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                
                keypoints.append(pose_label)
                all_data.append(keypoints)

# Save CSV
columns = [f'{coord}{i}' for i in range(33) for coord in ['x', 'y', 'z']]
columns.append('label')
df = pd.DataFrame(all_data, columns=columns)
df.to_csv('yoga_keypoints.csv', index=False)
print("\n✅ yoga_keypoints.csv saved.")
