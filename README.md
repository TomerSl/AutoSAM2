# AutoSAM2 for Medical Video Segmentation

This project adapts the Segment Anything Model 2 (SAM2) for medical video segmentation of polyp sequences, by using the AutoSAM approach. It replaces the prompt encoder in SAM2 with the custom prompt encoder from AutoSAM that's better suited for medical image segmentation tasks.

## Project Structure

- `datasets/` - Dataset handling code
  - `polyp_gen.py` - Implementation for the polyp_gen video dataset
  - `__init__.py` - Dataset factory

- `train.py` - Training script for video segmentation
- `inference.py` - Evaluation and inference script

## Dataset

The `polyp_gen` dataset is structured as a video dataset with sequential frames. The structure is:

```
polyp_gen/
├── train/
│   ├── images/
│   │   ├── sequence1/
│   │   │   ├── frame1.jpg
│   │   │   ├── frame2.jpg
│   │   │   └── ...
│   │   ├── sequence2/
│   │   └── ...
│   ├── masks/
│   │   ├── sequence1/
│   │   │   ├── frame1.jpg
│   │   │   ├── frame2.jpg
│   │   │   └── ...
│   │   ├── sequence2/
│   │   └── ...
│   └── train_list.txt (contains sequence names)
└── valid/
    ├── images/
    │   └── ...
    ├── masks/
    │   └── ...
    └── val_list.txt (contains sequence names)
```

## Key Features

1. **Custom Prompt Encoder**: Replaces SAM2's default prompt encoder with the one from AutoSAM, optimized for medical image segmentation.
2. **Video Segmentation**: Leverages SAM2's ability to handle video sequences, maintaining temporal consistency in polyp segmentation through proper handling of video frames.
3. **VideoDatapoint Structure**: Uses a structured video data representation with frames and objects to efficiently process and track polyps across frames.
4. **Focal Loss**: Uses focal loss for better handling of class imbalance in medical segmentation tasks.

## Video Processing Capabilities

The implementation uses VideoDatapoint, Frame, and Object classes to represent video data:

- **VideoDatapoint**: Contains a sequence of frames from a video and metadata
- **Frame**: Represents a single frame with multiple potential objects
- **Object**: Represents a segmented object (polyp) within a frame

This structure allows for:
- Efficient tracking of objects across frames
- Temporal consistency in segmentation
- Future extension to multi-object tracking

## Usage

### Setup

Make sure you have a compatible version of PyTorch installed. This implementation requires:
- Python 3.8+
- PyTorch 1.13+
- CUDA 11.6+ (for GPU acceleration)

### Training

To train the model on video data:

```bash
python train.py --checkpoint path/to/sam2_checkpoint.pth --batch-size 8 --epochs 50 --lr 1e-4
```

Additional options:
- `--dataset`: Dataset name (default: 'polyp_gen')
- `--gpu`: GPU ID to use (default: 0)
- `--cpu`: Use CPU instead of GPU
- `--output-dir`: Directory to save results (default: 'output')

### Evaluation

To evaluate a trained model on video sequences:

```bash
python inference.py --checkpoint path/to/sam2_checkpoint.pth --weights path/to/trained_weights.pth
```

Additional options:
- `--dataset`: Dataset name (default: 'polyp_gen')
- `--batch-size`: Batch size for evaluation (default: 8)
- `--output-dir`: Directory to save results (default: 'results')

## Implementation Details

### SAM2 Modification

The original SAM2 model has been modified by replacing the prompt encoder with AutoSAM's version. This implementation is specifically designed to work with video sequences of medical polyp data.

Key changes:
1. The prompt encoder from AutoSAM has been adapted to work with SAM2's architecture for video processing.
2. The data loading pipeline supports sequential frames from video sequences.
3. Batches are prepared to maintain temporal information between frames.

### Loss Functions

The implementation uses a combination of:
- Focal Loss: For better handling of class imbalance
- Dice Loss: As a region-based loss for segmentation quality

### Performance Considerations

For optimal performance with video data:
- Use a GPU with at least 8GB VRAM
- Increase batch size for smoother convergence when resources allow
- When processing long sequences, consider sampling to avoid memory issues

## Acknowledgements

This project is built upon:
- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI Research
- [AutoSAM](https://github.com/imed-lab) for medical segmentation approach

## License

This project follows the licensing terms of the original SAM2 repository. 