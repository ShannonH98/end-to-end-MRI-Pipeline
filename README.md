# End-to-End MRI Pipeline
This project is to demonstrate understanding from implementing a complete pipe for processing 3D brain MRI scans in .nii.gz format. It includes skull stripping using FSL, extraction of 2D slices from volumetric data, preprocessing and normalization, and preparation of data for deep learning models.

## Pipeline stages

| Stage | Script | Description |
|---|---|---|
| 1. Skull stripping | `preprocessing/fsl_preprocessing.py` | Removes non-brain tissue from raw `.nii` files using FSL `bet` |
| 2. Slice extraction | `preprocessing/slice_extraction.py` | Extracts 2D axial slices from each 3D volume, saves as `.png` |
| 3. Dataset loading | `preprocessing/dataset.py` | Loads slices into a labelled PyTorch `Dataset` (HC=0, AVH+=1) |
| 4. Model training | `train.py` | Trains a 3-layer CNN (`models/cnn.py`) with 80/20 train/val split |
| 5. Visualisation | `visualize.py` | Plots loss, accuracy, confusion matrix, sample slices, and Grad-CAM |

## Data

- `data/raw/` — 10 raw T1-weighted `.nii` scans (5 HC, 5 AVH+)
- `data/processed/` — skull-stripped volumes (`_brain.nii.gz`)
- `data/slices/` — extracted axial PNG slices per subject
- `data/participants.tsv` — subject metadata (age, sex, IQ, group, PSYRATS score)

## Usage

**Run the full pipeline from the command line:**

```bash
python run_pipeline.py   # skull strip + slice extraction
python train.py          # train the CNN
python visualize.py      # generate plots
```

**Or run interactively in Jupyter:**

```bash
jupyter notebook pipeline_runnable.ipynb
```

The runnable notebook executes every stage in order with inline outputs and visualisations.

## Model

`BrainCNN` — a lightweight 3-layer convolutional network:
- Input: 1×128×128 greyscale slice
- Three Conv→ReLU→MaxPool blocks (16→32→64 channels)
- Fully connected: 64×16×16 → 128 → 2 (HC / AVH+)

## Requirements

- Python 3.10+
- FSL (`bet` must be on PATH for skull stripping)
- `nibabel`, `Pillow`, `numpy`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`
