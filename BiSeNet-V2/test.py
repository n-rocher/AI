import os

SEQUENCE_SIZE = 3
MAX_TIME_JUMP = 100

dataset_folder = r"F:\\A2D2 Camera Semantic\\"

data_image = []
data_label = []
possible_batch = []

for data_type in ["training", "validation"]:

    current_folder = os.path.join(dataset_folder, data_type)
    camera_day_folders = [os.path.join(current_folder, item) for item in os.listdir(current_folder) if os.path.isdir(os.path.join(current_folder, item))]

    day_data = []

    for folder in camera_day_folders:
        camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
        label_files_folder = os.path.join(folder, "label", "cam_front_center")

        camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
        label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]
      
        camera_files_id = list(sorted([int(file[-13:-4]) for file in os.listdir(camera_files_folder)]))

        assert len(camera_files_files) == len(label_files_files), "Hum les tailles sont différentes entre camera et label pour " + camera_day_folders

        # On boucle sur touts les id d'image
        for i in range(0, len(camera_files_files) - SEQUENCE_SIZE):

            result = True
            for j in range(1, SEQUENCE_SIZE): # On verifie si les images de i à i+j n'ont pas un écart de temps > à MAX_TIME_JUMP
                result = result and (camera_files_id[i+j] - camera_files_id[i+j-1]) <= MAX_TIME_JUMP
                if result is not True:
                    break
            
            # Si écart de temps ok entre toutes les images, on ajoute dans le tableau
            if result:
                possible_batch.append([camera_files_files[i:i+SEQUENCE_SIZE], label_files_files[i+SEQUENCE_SIZE-1]])

print(len(possible_batch), possible_batch[0])