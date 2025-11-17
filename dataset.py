import os
import zipfile
import shutil
import glob
import random

# Path where ZIP files are located
data_dir = r'C:\xamppnew\htdocs\EmoGame\EmoGame\Newdata'  
zip_files = [
    'archive (3).zip',
    'archive.zip',
    'archive (4).zip'
]

# Folder to extract all datasets
extract_dir = 'data/raw'
os.makedirs(extract_dir, exist_ok=True)

# Step 1: Extract ZIP files
for zip_file in zip_files:
    zip_path = os.path.join(data_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
print("All ZIP files extracted.")

# Step 2: Create Keras-friendly folder structure
train_dir = 'data/train'
val_dir = 'data/test'
emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

for folder in [train_dir, val_dir]:
    for emotion in emotions:
        os.makedirs(os.path.join(folder, emotion), exist_ok=True)

# Step 3: Combine all datasets and split into train/test
random.seed(42)

for dataset_folder in os.listdir(extract_dir):
    dataset_path = os.path.join(extract_dir, dataset_folder)
    
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_path):
            continue  # skip if this emotion does not exist in this dataset
        
        img_files = glob.glob(os.path.join(emotion_path, '*.*'))
        random.shuffle(img_files)
        
        split = int(0.8 * len(img_files))
        train_files = img_files[:split]
        test_files = img_files[split:]
        
        for f in train_files:
            shutil.copy(f, os.path.join(train_dir, emotion))
        for f in test_files:
            shutil.copy(f, os.path.join(val_dir, emotion))

print("Datasets combined, shuffled, and split into train/test folders.")
