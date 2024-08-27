import argparse
import xml.etree.ElementTree as ET
import os
from glob import glob

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(input_dir, output_dir, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = os.path.join(input_dir, 'annotations', basename_no_ext + '.xml')
    if not os.path.exists(in_file):
        print(f"Warning: Annotation file not found for {basename}")
        return

    out_file = os.path.join(output_dir, basename_no_ext + '.txt')
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(out_file, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if difficult is not None:
                difficult = int(difficult.text)
            else:
                difficult = 0

            cls = obj.find('name').text
            if cls not in classes or difficult == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_bbox((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    print(f"Processed: {basename}")






def read_classes(input_dir):
    classes_file = os.path.join(input_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PascalVOC annotations to YOLO format')
    parser.add_argument('input_dir', type=str, help='Path to input directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    try:
        classes = read_classes(args.input_dir)
        print(f"Classes found: {classes}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Look for 'annotations' and 'images' subdirectories
    annotations_dir = os.path.join(args.input_dir, 'annotations')
    images_dir = os.path.join(args.input_dir, 'images')
    if not os.path.exists(annotations_dir) or not os.path.exists(images_dir):
        print(f"Error: 'annotations' or 'images' subdirectory not found in {args.input_dir}")
        exit(1)

    # Find all XML files in the 'annotations' subdirectory
    xml_files = glob(os.path.join(annotations_dir, '*.xml'))
    print(f"Number of XML files found: {len(xml_files)}")
    
    for xml_file in xml_files:
        try:
            # Construct the corresponding image path
            image_name = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
            image_path = os.path.join(images_dir, image_name)
            
            if os.path.exists(image_path):
                convert_annotation(args.input_dir, args.output_dir, image_path)
            else:
                print(f"Warning: Image file not found for {xml_file}")
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")

    print("Conversion completed.")
