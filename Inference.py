
import argparse
import cv2
import os
from ultralytics import YOLO
import numpy as np

def draw_boxes(img, boxes, labels, scores, color=(0, 255, 0)):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def process_image(image_path, person_model, ppe_model, output_dir):
    img = cv2.imread(image_path)
    original_img = img.copy()

    # Person detection
    results = person_model(img)
    person_boxes = results[0].boxes.xyxy.cpu().numpy()
    person_scores = results[0].boxes.conf.cpu().numpy()

    # Draw person bounding boxes
    img = draw_boxes(img, person_boxes, ['Person'] * len(person_boxes), person_scores)

    # PPE detection on cropped person images
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_img = original_img[y1:y2, x1:x2]
        
        ppe_results = ppe_model(cropped_img)
        ppe_boxes = ppe_results[0].boxes.xyxy.cpu().numpy()
        ppe_labels = [ppe_model.names[int(c)] for c in ppe_results[0].boxes.cls.cpu().numpy()]
        ppe_scores = ppe_results[0].boxes.conf.cpu().numpy()

        # Adjust PPE bounding boxes to original image coordinates
        adjusted_ppe_boxes = ppe_boxes.copy()
        adjusted_ppe_boxes[:, [0, 2]] += x1
        adjusted_ppe_boxes[:, [1, 3]] += y1

        # Draw PPE bounding boxes
        img = draw_boxes(img, adjusted_ppe_boxes, ppe_labels, ppe_scores, color=(255, 0, 0))

    # Save the output image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

def main(input_dir, output_dir, person_det_model, ppe_detection_model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    person_model = YOLO(person_det_model)
    ppe_model = YOLO(ppe_detection_model)

    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            process_image(image_path, person_model, ppe_model, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference on images using person and PPE detection models')
    parser.add_argument('input_dir', type=str, help='Path to input image directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('person_det_model', type=str, help='Path to person detection model weights')
    parser.add_argument('ppe_detection_model', type=str, help='Path to PPE detection model weights')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)