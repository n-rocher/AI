import os

input_dir = "images/100k/"
target_dir = "labels/drivable/colormaps/"
img_size = (1280, 720)
num_classes = 3
batch_size = 32

input_train_img_paths = sorted([os.path.join(input_dir, "train/", fname) for fname in os.listdir(input_dir + "train/") if fname.endswith(".jpg")])

target_train_img_paths = sorted([os.path.join(target_dir, "train/", fname) for fname in os.listdir(target_dir + "train/") if fname.endswith(".png")])

print("Number of samples:", len(input_train_img_paths))

for input_path, target_path in zip(input_train_img_paths[:10], target_train_img_paths[:10]):
    print(input_path, "|", target_path)