import os
import shutil

def organize_data(source_dir, target_dir):
    # Define the subdirectories for train, val, and test
    splits = ['train', 'test']
    
    # Ensure target directories exist
    for split in splits:
        for category in ['NORMAL', 'PNEUMONIA']:
            os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)
    
    # Move data from source to target
    for split in splits:
        for category in ['NORMAL', 'PNEUMONIA']:
            src_dir = os.path.join(source_dir, split, category)
            dest_dir = os.path.join(target_dir, split, category)
            
            if os.path.exists(src_dir):
                # Move files from source to destination
                for file_name in os.listdir(src_dir):
                    shutil.move(os.path.join(src_dir, file_name), os.path.join(dest_dir, file_name))

    # Optionally create a validation split if needed
    create_validation_split(target_dir, val_split=0.2)

def create_validation_split(data_dir, val_split=0.2):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Create validation directories if they don't exist
    os.makedirs(val_dir, exist_ok=True)
    for class_name in ['NORMAL', 'PNEUMONIA']:
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Get list of files for each class
        files = os.listdir(os.path.join(train_dir, class_name))
        random.shuffle(files)
        val_count = int(len(files) * val_split)

        # Move files to validation folder
        for file_name in files[:val_count]:
            shutil.move(
                os.path.join(train_dir, class_name, file_name),
                os.path.join(val_dir, class_name, file_name)
            )

if __name__ == '__main__':
    # Define source and target directories
    source_directory = 'data/chest_xray/'
    target_directory = 'data/'

    # Organize the data
    organize_data(source_directory, target_directory)
