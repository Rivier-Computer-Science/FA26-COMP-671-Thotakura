# Traffic Sign Recognition Using CNNs and ResNet-18

This deep learning project recognizes 43 classes of German traffic
signs from road images. It compares a convolutional neural network
trained from scratch with an ImageNet-pretrained ResNet-18.

## Project Goals

- Build a reproducible GTSRB image-classification pipeline.
- Compare a baseline CNN with fine-tuned ResNet-18.
- Measure accuracy and macro F1-score.
- Evaluate the effect of image augmentation.
- Analyze difficult classes and incorrect predictions.
- Demonstrate inference on a single traffic-sign image.

## Dataset

The project uses the German Traffic Sign Recognition Benchmark
(GTSRB) provided by `torchvision`.

- Training images: 26,640
- Official test images: 12,630
- Number of classes: 43
- Validation fraction: 15% of the training data
- Input size: 64 x 64 RGB

The dataset is downloaded automatically into `data/`. This directory
is ignored by Git and must not be committed.

## Results

| Experiment | Test accuracy | Macro F1 | Best epoch |
| --- | ---: | ---: | ---: |
| Baseline CNN | 93.18% | 90.58% | 15 |
| ResNet-18 without augmentation | 95.54% | 93.12% | 6 |
| Fine-tuned ResNet-18 with augmentation | **97.03%** | **95.08%** | 9 |

Fine-tuned ResNet-18 improved accuracy by approximately 3.85
percentage points over the baseline CNN. Augmentation improved
ResNet-18 accuracy by approximately 1.49 percentage points.

## Requirements

- Python 3.11
- PyTorch
- torchvision
- pandas
- scikit-learn
- matplotlib
- seaborn
- Pillow
- PyYAML
- tqdm
- pytest and Ruff for development

## Environment Setup

Clone the assigned repository:

```bash
git clone https://github.com/Rivier-Computer-Science/FA26-COMP-671-Thotakura.git
cd FA26-COMP-671-Thotakura
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements-dev.txt
```

Verify the installation:

```bash
python -m pytest -v
```

## Dataset Exploration

Download and explore GTSRB:

```bash
python -m src.traffic_sign_recognition.explore
```

This creates:

```text
artifacts/metrics/dataset_summary.json
artifacts/metrics/class_distribution.csv
artifacts/figures/class_distribution.png
artifacts/figures/sample_images.png
```

## Train the Baseline CNN

```bash
python -m src.traffic_sign_recognition.train \
  --config configs/baseline.yaml
```

## Train Fine-Tuned ResNet-18

```bash
python -m src.traffic_sign_recognition.train \
  --config configs/resnet18.yaml
```

## Run the Augmentation Ablation

```bash
python -m src.traffic_sign_recognition.train \
  --config configs/resnet18_no_augmentation.yaml
```

Each run saves its metrics under:

```text
artifacts/metrics/<experiment_name>/
```

Generated files include:

- `config.json`
- `history.csv`
- `summary.json`
- `predictions.csv`
- `classification_report.csv`
- `confusion_matrix.png`

The best checkpoint is saved under `artifacts/checkpoints/`. Checkpoints are intentionally ignored by Git because they are large.

## Compare Experiments

After all experiments finish, run:

```bash
python -m src.traffic_sign_recognition.compare
```

This creates:

```text
artifacts/figures/model_comparison.csv
artifacts/figures/model_comparison.png
```

## Failure Analysis

Analyze the strongest model:

```bash
python -m src.traffic_sign_recognition.failure_analysis \
  --predictions artifacts/metrics/resnet18_finetuned/predictions.csv \
  --output-dir artifacts/analysis/resnet18_finetuned
```

The analysis produces:

- Incorrect predictions
- Difficult classes
- Frequent confusion pairs
- High-confidence error counts
- A difficult-class visualization

ResNet-18 made 375 incorrect predictions out of 12,630 test examples. Of those mistakes, 81 had confidence of at least 90%.

## Single-Image Inference

After training ResNet-18, classify one image:

```bash
python -m src.traffic_sign_recognition.predict \
  data/gtsrb/GTSRB/Final_Test/Images/06578.ppm \
  --checkpoint artifacts/checkpoints/resnet18_finetuned.pt \
  --top-k 3
```

Example output:

```text
1. No passing (class 9): 99.98%
2. No passing for vehicles over 3.5 tons (class 10): 0.00%
3. Vehicles over 3.5 tons prohibited (class 16): 0.00%
```

## Repository Structure

```text
configs/                         Experiment configurations
src/traffic_sign_recognition/    Project source code
tests/                           Automated tests
artifacts/metrics/               Evaluation metrics and predictions
artifacts/figures/               Report and presentation figures
artifacts/analysis/              Failure-analysis results
artifacts/checkpoints/           Local model checkpoints
report/                          Written report
presentation/                    Presentation materials
```

## Reproducibility

- All experiments use random seed 42.
- Train and validation indices are deterministic.
- The official GTSRB test split is not used for training.
- Augmentation is applied only to training images.
- Validation and test transforms are deterministic.
- Experiment settings are stored in YAML files.
- Best checkpoints are selected using validation loss.
- Early stopping prevents unnecessary training.

Hardware and operating-system differences can cause small numerical differences in repeated experiments.

## Limitations and Responsible Use

GTSRB primarily represents German signs and does not cover every country, weather condition, camera, occlusion, or damaged sign. High benchmark accuracy does not establish that this model is safe for deployment.

This project is an educational prototype. It must not be used as the only perception or decision component in a real vehicle.

## Author

Bindu Thotakura  
COMP-671 Deep Learning Final Project