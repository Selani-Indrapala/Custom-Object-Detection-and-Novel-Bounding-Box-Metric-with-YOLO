import os
import yaml
import argparse
from glob import glob
from shutil import copytree
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

class DatasetConverter:
    def __init__(self, annotations_dir, images_dir, output_dir, class_mapping):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.class_mapping = class_mapping

        self.text_labels_dir = os.path.join(output_dir, 'labels')
        self.imgs_dir = os.path.join(output_dir, 'images')
        
        os.makedirs(self.text_labels_dir, exist_ok=True)
        os.makedirs(self.imgs_dir, exist_ok=True)

    def convert_xml_to_yolo(self, xml_file):
        with open(xml_file, 'r') as f:
            data = f.read()
            soup = BeautifulSoup(data, 'xml')  # Parse XML annotation file

            # Extract image width and height
            img_size = soup.find('size')
            img_width = int(img_size.find('width').text)
            img_height = int(img_size.find('height').text)

            # Find all object annotations in the image
            objects = soup.find_all('object')
            obj_list = []

            # Process each object in the image
            for obj in objects:
                label = self.class_mapping(obj.find('name').text)  # Get class ID
                xmin = int(obj.find('xmin').text)
                ymin = int(obj.find('ymin').text)
                xmax = int(obj.find('xmax').text)
                ymax = int(obj.find('ymax').text)

                # Convert bounding box to YOLO format
                x = ((xmin + xmax) / 2) / img_width  # Center X (normalized)
                y = ((ymin + ymax) / 2) / img_height  # Center Y (normalized)
                width = (xmax - xmin) / img_width  # Width (normalized)
                height = (ymax - ymin) / img_height  # Height (normalized)

                obj_list.append([label, x, y, width, height])

            return obj_list

    def save_yolo_labels(self, label_list, filename):
        txt_label_path = os.path.join(self.text_labels_dir, filename)
        with open(txt_label_path, 'w') as f:
            for obj in label_list:
                f.write(f"{obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]}\n")  # Write YOLO label format

    def process_annotations(self):
        xml_files = sorted(glob(os.path.join(self.annotations_dir, '*.xml')))
        
        for xml_file in xml_files:
            label_list = self.convert_xml_to_yolo(xml_file)
            file_name = os.path.basename(xml_file).replace('.xml', '.txt')
            self.save_yolo_labels(label_list, file_name)

    def copy_images(self):
        copytree(self.images_dir, self.imgs_dir)

class DatasetSplitter:
    def __init__(self, images_dir, output_dir):
        self.images_dir = images_dir
        self.output_dir = output_dir

    def split_dataset(self):
        img_list = glob(self.images_dir + '/*.png')

        # Split into training (80%) and temp (20%)
        train_img, val_img = train_test_split(img_list, test_size=0.2, random_state=0)

        return train_img, val_img

    def save_image_paths(self, train_img, val_img):
        with open(os.path.join(self.output_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_img) + '\n')
        with open(os.path.join(self.output_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(val_img) + '\n')

    def update_yaml(self):
        data = {
            'train': 'dataset/train.txt',  # Training images list (80%)
            'val': 'dataset/val.txt',  # Validation images list (10%)
            'nc': 2,  # Number of classes (cat, dog)
            'names': ['cat', 'dog']  # Class labels
        }

        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(data, f)

class DogCatDatasetProcessor:
    def __init__(self, annotations_dir, images_dir, output_dir, class_mapping):
        self.converter = DatasetConverter(annotations_dir, images_dir, output_dir, class_mapping)
        self.splitter = DatasetSplitter(os.path.join(output_dir, 'images'), output_dir)

    def process(self):
        self.converter.process_annotations()
        self.converter.copy_images()
        
        train_img, val_img = self.splitter.split_dataset()
        self.splitter.save_image_paths(train_img, val_img)
        self.splitter.update_yaml()

def main(args):
    class_mapping = lambda x: 0 if x == 'cat' else 1  # 'cat' -> 0, 'dog' -> 1

    processor = DogCatDatasetProcessor(args.annotations_dir, args.images_dir, args.output_dir, class_mapping)
    processor.process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process Dog and Cat detection dataset into YOLO format")

    # Add command-line arguments
    parser.add_argument('-annotations_dir', type=str, required=True, help="Path to the annotations directory")
    parser.add_argument('-images_dir', type=str, required=True, help="Path to the images directory")
    parser.add_argument('-output_dir', type=str, required=True, help="Path to store the output dataset")
    
    args = parser.parse_args()
    main(args)
