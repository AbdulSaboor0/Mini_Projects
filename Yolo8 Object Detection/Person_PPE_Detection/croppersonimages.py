import cv2
import os
from glob import glob
import numpy as np

def crop_and_adjust_annotations(input_dir, output_dir):
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    image_paths = glob(os.path.join(input_dir, '*.jpg'))
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        h, w = img.shape[:2]
        
        basename = os.path.basename(image_path)
        basename_no_ext = os.path.splitext(basename)[0]
        
        annotation_path = os.path.join(os.path.dirname(input_dir), 'labels', basename_no_ext + '.txt')
        print(f"Looking for annotation file: {annotation_path}")
        
        if not os.path.exists(annotation_path):
            print(f"Annotation file not found: {annotation_path}")
            continue

        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        person_boxes = []
        for line in lines:
            class_id, x, y, width, height = map(float, line.strip().split())
            if class_id == 0:  # Assuming 0 is the class ID for person
                x1 = int((x - width/2) * w)
                y1 = int((y - height/2) * h)
                x2 = int((x + width/2) * w)
                y2 = int((y + height/2) * h)
                person_boxes.append((x1, y1, x2, y2))
        
        print(f"Found {len(person_boxes)} person boxes")

        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            cropped_img = img[y1:y2, x1:x2]
            cropped_h, cropped_w = cropped_img.shape[:2]
            
            output_image_path = os.path.join(output_dir, 'images')
            os.makedirs(output_image_path, exist_ok=True)
            output_image_path = os.path.join(output_image_path, f"{basename_no_ext}_person{i}.jpg")
            cv2.imwrite(output_image_path, cropped_img)
            print(f"Saved cropped image: {output_image_path}")
            
            output_annotation_path = os.path.join(output_dir, 'labels')
            os.makedirs(output_annotation_path, exist_ok=True)
            output_annotation_path = os.path.join(output_annotation_path, f"{basename_no_ext}_person{i}.txt")
            with open(output_annotation_path, 'w') as f:
                for line in lines:
                    class_id, x, y, width, height = map(float, line.strip().split())
                    if class_id != 0:  # Skip person annotations
                        # Adjust coordinates relative to the cropped image
                        new_x = (x * w - x1) / cropped_w
                        new_y = (y * h - y1) / cropped_h
                        new_width = (width * w) / cropped_w
                        new_height = (height * h) / cropped_h
                        
                        # Ensure coordinates are within bounds
                        new_x = max(0, min(1, new_x))
                        new_y = max(0, min(1, new_y))
                        new_width = max(0, min(1, new_width))
                        new_height = max(0, min(1, new_height))
                        
                        f.write(f"{int(class_id)} {new_x} {new_y} {new_width} {new_height}\n")
            print(f"Saved annotation file: {output_annotation_path}")

# Usage
input_dir = r'C:\Users\saboo\Downloads\datasets\images'
output_dir = r'C:\Users\saboo\Downloads\Syook Dataset-20240825T074635Z-001\Syook Dataset\datasets\CROPPED'

print("Starting processing...")
crop_and_adjust_annotations(input_dir, output_dir)
print("Processing completed.")
