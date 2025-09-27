import json
import os
import shutil
from tqdm import tqdm

# --- Configuration ---
# The main folder containing your train, valid, and test sets
DATASET_DIR = '/Users/lakshmanmandapati/Downloads/AI SPARK/archive'

# --- Main Script ---

def convert_coco_to_yolo(source_dir):
    """
    Converts a dataset from COCO JSON format to YOLO TXT format.
    Also organizes files into 'images' and 'labels' subdirectories.
    """
    json_path = os.path.join(source_dir, '_annotations.coco.json')
    
    if not os.path.exists(json_path):
        print(f"⏩ Skipping '{source_dir}' as no JSON annotation file was found.")
        return

    print(f"🔄 Processing '{source_dir}'...")

    # Create destination folders
    images_dest = os.path.join(source_dir, 'images')
    labels_dest = os.path.join(source_dir, 'labels')
    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)

    # Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Create mappings
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Process annotations
    annotations_by_image = {}
    for ann in tqdm(coco_data['annotations'], desc=f"  -> Converting annotations in {os.path.basename(source_dir)}"):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox'] # [x_min, y_min, width, height]
        
        img_info = images_info[image_id]
        img_w = img_info['width']
        img_h = img_info['height']

        # Convert COCO bbox to YOLO format
        x_center = (bbox[0] + bbox[2] / 2) / img_w
        y_center = (bbox[1] + bbox[3] / 2) / img_h
        width = bbox[2] / img_w
        height = bbox[3] / img_h
        
        yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        filename = img_info['file_name']
        if filename not in annotations_by_image:
            annotations_by_image[filename] = []
        annotations_by_image[filename].append(yolo_line)

    # Write YOLO label files
    for filename, lines in annotations_by_image.items():
        label_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(labels_dest, label_filename), 'w') as f:
            f.writelines(lines)
            
    # Move images to the 'images' subdirectory
    print(f"  -> Organizing image files...")
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in image_files:
        shutil.move(os.path.join(source_dir, img_file), os.path.join(images_dest, img_file))

    # Optional: remove the original annotation file after processing
    os.remove(json_path)
    
    print(f"✅ Finished processing '{source_dir}'.")


if __name__ == '__main__':
    subfolders = ['train', 'valid', 'test']
    for subfolder in subfolders:
        path = os.path.join(DATASET_DIR, subfolder)
        if os.path.isdir(path):
            convert_coco_to_yolo(path)
    print("\n🎉 All datasets converted successfully!")