import os

train_dir = 'data/train'
test_dir = 'data/test'

emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print("=== Dataset Summary ===")
for split_dir, name in [(train_dir, "Train"), (test_dir, "Test")]:
    print(f"\n{name} set:")
    total = 0
    for emotion in emotions:
        folder = os.path.join(split_dir, emotion)
        count = len(os.listdir(folder))
        print(f"  {emotion}: {count} images")
        total += count
    print(f"  >> Total: {total} images")

