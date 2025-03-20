# AutoSAM2: Adapting SAM2 to Medical Videos

AutoSAM2 is a framework that adapts Meta's Segment Anything Model v2 (SAM2) for fully automated segmentation in medical videos—specifically, for polyp segmentation in colonoscopy data. By replacing SAM2's interactive prompt encoder with a learnable, automatic prompt encoder, AutoSAM2 generates segmentation prompts directly from the image. This approach leverages SAM2's powerful image encoder, memory mechanisms, and mask decoder while fine-tuning a new module tailored to the polyp segmentation task.

**Note:** In our evaluation, we perform segmentation in image prediction mode rather than video mode. Additionally, we provide 10 example evaluation images to quickly assess the model's performance.

## Table of Contents
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Downloading SAM2 Weights](#downloading-sam2-weights)
- [Downloading and Organizing the PolypGen Dataset](#downloading-and-organizing-the-polypgen-dataset)
- [Training the Model](#training-the-model)
- [Evaluation and Inference](#evaluation-and-inference)
- [Example Evaluation Images](#example-evaluation-images)


## Installation and Environment Setup

### Clone the Repository:
```bash
git clone https://github.com/TomerSl/AutoSAM2.git
cd AutoSAM2
```

### Activate the Virtual Environment:
Create a virtual environment using Python's built-in venv (or your favorite tool) and activate it:
```bash
python3 -m venv sam2_env
source sam2_env/bin/activate
```

### Install Dependencies:
Install the required packages:
```bash
pip install -r requirements.txt
```


## Downloading SAM2 Weights

### Download the Weights:
Download the SAM2 checkpoint (e.g., `sam2.1_hiera_large.pt`) from Meta's official repository or the provided link or run the bash command in sam2-main/checkpoints/.


### Place the Checkpoint:
Place the downloaded checkpoint in the `sam2-main/checkpoints/` directory.

### Configuration Files:
Ensure that the configuration file (e.g., `sam2.1_hiera_l.yaml`) is available in the `sam2-main/configs/` directory.

## Downloading and Organizing the PolypGen Dataset

### Download the Data:
Download the PolypGen dataset from its official source. 

### Directory Structure:
Organize the data as follows:
```
polyp_gen/
  ├── train/
  │     ├── images/
  │     │     ├── <sequence_id>/
  │     │     │     ├── 0001.jpg
  │     │     │     ├── 0002.jpg
  │     │     │     └── ...
  │     ├── masks/
  │     │     ├── <sequence_id>/
  │     │     │     ├── 0001.jpg
  │     │     │     ├── 0002.jpg
  │     │     │     └── ...
  │     └── train_list.txt
  └── valid/
        ├── images/
        │     ├── <sequence_id>/
        │     │     ├── 0001.jpg
        │     │     ├── 0002.jpg
        │     │     └── ...
        ├── masks/
        │     ├── <sequence_id>/
        │     │     ├── 0001.jpg
        │     │     ├── 0002.jpg
        │     │     └── ...
        └── val_list.txt
```

### File Lists:
Ensure that `train_list.txt` and `val_list.txt` list the sequence IDs (one per line) corresponding to the subdirectories in the images and masks folders.

## Training the Model

### (Optional) Configure Data Augmentations:
You can edit the `get_transforms` function in `dataset/polyp_gen.py` to enable augmentations (such as color jitter, rotation, and horizontal flip). For debugging, you can disable them by returning `None`.

### Run Training:
```bash
python train.py --data_dir /home/fodl/tomerslor/AutoSAM2/polyp_gen --output_dir output --batch_size 12 --epochs 200 --lr 5e-5
```

This script will:
- Load and preprocess the training and validation data.
- Load the SAM2 models from the `sam2-main` directory.
- Train the ModelEmb module using a combined Dice and BCE loss.
- Save checkpoints and training visualizations to the output directory.

## Evaluation and Inference

After training, evaluate your model and run inference using `inference.py` (which reuses the evaluation logic from training). This script operates in image prediction mode.

### Run Inference:
```bash
python inference.py  --pretrained /home/fodl/tomerslor/AutoSAM2/output/checkpoints/model_epoch_82.pth
```

The script will:
- Load the validation dataset.
- Load the trained ModelEmb along with its pretrained weights.
- Load the SAM2 image model.
- Evaluate the model on the validation set (using the same evaluation logic as in training).
- Generate and save segmentation visualizations for all frames in the specified video (10 example images are produced for evaluation).

**Note:** The `--pretrained` argument accepts either a full path or a relative path (e.g., `output/checkpoints/model_epoch_82.pth`), as long as the file is accessible from your working directory.

## Example Evaluation Images

During inference, 10 example evaluation images are generated. For each example, the following are saved:
- The original input image.
- The ground truth segmentation mask.
- The predicted segmentation mask.

These images are saved in `output/visualizations/` (or a similar directory) to help you quickly assess model performance.


## Acknowledgements

This project is built upon:
- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI Research
- [AutoSAM](https://github.com/talshaharabany/AutoSAM) for medical segmentation approach

## License

This project follows the licensing terms of the original SAM2 repository. 
