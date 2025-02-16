
# Custom Object Detection and Novel Bounding Box Metric with YOLO

This project demonstrates how to preprocess data and train a custom YOLOv5 model to detect cats and dogs in images, including the steps to preprocess annotations, train, and evaluate the model.

## Prerequisites

Make sure you have the following installed:
- Python 3.7 or higher
- Git
- pip (Python package manager)

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Selani-Indrapala/Custom-Object-Detection-and-Novel-Bounding-Box-Metric-with-YOLO.git
cd Custom-Object-Detection-and-Novel-Bounding-Box-Metric-with-YOLO
pip install -r requirements.txt
```

## Data Setup

The dataset used in this project consists of images and annotations for detecting cats and dogs. You'll need to specify the paths for the annotations, images, and output directories when running the preprocessing script.

### 1. Preprocess Data

The preprocessing script converts XML annotations to the YOLO format and prepares the dataset for training. To run the script, use the following command:

```bash
# Set the paths to your annotations, images, and output directory
annotations_path = '/kaggle/input/dog-and-cat-detection/annotations'
images_path = '/kaggle/input/dog-and-cat-detection/images'
output_path = 'dataset'

# Run the preprocessing script
python PreprocessData.py -annotations_dir $annotations_path -images_dir $images_path -output_dir $output_path
```

This will:
- Convert the XML annotations to YOLO format
- Copy the images into the output directory
- Create `train.txt` and `val.txt` for training and validation splits
- Generate `data.yaml` to configure the dataset for YOLOv5

## Training the Model

Once the data is preprocessed, you can train a YOLOv5 model on the dataset.

### 2. Train the Model

Run the following command to start training the YOLOv5 model:

```bash
python train.py   --img 416   --batch 16   --epochs 10   --data dataset/data.yaml   --cfg ./models/yolov5s.yaml   --weights ''   --name cat_and_dog_yolov5s_results
```

This will:
- Train the model with images resized to 416x416
- Use a batch size of 16
- Train for 10 epochs
- Save the model results in the `runs/train/cat_and_dog_yolov5s_results` directory

### 3. The Loss Function 

The current implementation is for the loss function that only considers the size similarity as an additional metric as this gave the best performance. The loss function can be viewed under utils/loss.py in the ComputeLoss class under the __call__ function. The hyperparameter weights for each loss metric can be found under data/hyps/hyp.scratch-low.yaml.

```# Compute Aspect Ratio Loss
par = pwh[:, 0] / (pwh[:, 1] + 1e-6)  # Predicted Aspect Ratio (Avoid division by zero)
tar = tbox[i][:, 2] / (tbox[i][:, 3] + 1e-6)  # Target Aspect Ratio
lar += torch.mean((par - tar) ** 2)  # MSE loss for aspect ratio
```
```# Compute Center Alignment Loss
pcx = pxy[:, 0]  # Predicted center x-coordinate
pcy = pxy[:, 1]  # Predicted center y-coordinate
tcx = tbox[i][:, 0]  # Target center x-coordinate
tcy = tbox[i][:, 1]  # Target center y-coordinate                
lcenter = torch.mean((pcx - tcx) ** 2 + (pcy - tcy) ** 2)  # MSE loss for center alignment
lcenter = lcenter.unsqueeze(0)
```
```# Compute the area similarity
pred_area = pwh[:, 0] * pwh[:, 1]  # Predicted area (width * height)
target_area = tbox[i][:, 2] * tbox[i][:, 3]  # Target area (width * height)
lsize += torch.mean((pred_area - target_area) ** 2)
```
## Model Evaluation

After training, you can evaluate the model on the validation set using the following command:

```bash
python val.py   --weights runs/train/cat_and_dog_yolov5s_results/weights/best.pt   --data dataset/data.yaml   --img 416   --batch 16
```

This will:
- Load the best weights (`best.pt`) from the training process
- Run the evaluation on the validation set and display the performance metrics

## Directory Structure

Here’s what the project directory will look like after running the preprocessing and training steps:

```
Custom-Object-Detection-and-Novel-Bounding-Box-Metric-with-YOLO/
  ├── dataset/
  │   ├── images/             # Copied images
  │   ├── labels/             # YOLO formatted labels
  │   ├── train.txt           # Training image paths
  │   ├── val.txt             # Validation image paths
  │   └── data.yaml           # Dataset configuration
  ├── models/                 # YOLOv5 models
  ├── runs/                   # Training and evaluation results
  ├── PreprocessData.py       # Data preprocessing script
  ├── train.py                # Training script
  ├── val.py                  # Evaluation script
  └── requirements.txt        # Required dependencies
```

## Notes

- The dataset used for this example is for detecting cats and dogs and uses the Dog and Cat Detection dataset on Kaggle (https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection)
- The model configuration used in this example is `yolov5s.yaml` (small version of YOLOv5).


