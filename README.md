# Conditional Sketch-to-Image GAN

This project implements a conditional GAN architecture for translating unpaired sketches to realistic images, conditioned on class labels. The implementation is based on the ISIC skin lesion dataset.

## Project Overview

The main components of this project are:
- A conditional GAN that takes unpaired sketches as input and generates realistic images
- A label embedding network that conditions the generation process
- A classifier to evaluate the realism and accuracy of the generated images

## Requirements

```
torch>=1.7.0
torchvision>=0.8.1
numpy>=1.19.2
pandas>=1.1.3
matplotlib>=3.3.2
pillow>=8.0.1
scipy>=1.5.2
wandb>=0.10.12
```

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the ISIC dataset which includes:
- Training images and their labels
- Training unpaired sketches
- Test images and their labels
- Test unpaired sketches

Download the dataset from [this link](https://drive.google.com/drive/folders/1vYv5SmA6nu4PKB_5PIk6FoTCtyEWztKP?usp=sharing) and place the files in the `data` directory.

### Data Preprocessing

The dataset is preprocessed using the following transformations:
- Resize to 128x128 pixels
- Center crop to 128x128 pixels
- Normalize with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]

## Model Architecture

### Generator

The Generator has an encoder-decoder architecture:
- Encoder: Consists of 5 convolutional layers with ReLU activation and batch normalization
- Decoder: Consists of 5 transposed convolutional layers with ReLU activation and batch normalization
- Final layer uses Tanh activation to produce images with values in [-1, 1]

### Discriminator

The Discriminator evaluates the authenticity of generated images:
- 3 convolutional layers with LeakyReLU activation
- Batch normalization and dropout for regularization
- Final sigmoid activation for binary classification

### Label Embedding

The Label Embedding network transforms class labels into spatial feature maps:
- 3 fully connected layers with ReLU activation
- Reshapes output to 128x128 spatial dimension
- Concatenated with sketch input to condition the generation process

### Classifier

The Classifier evaluates the class accuracy of generated images:
- 5 convolutional layers with ReLU activation
- Batch normalization, max pooling, and dropout for regularization
- Fully connected layers for classification into 7 classes

## Usage

### Training

To train the model:

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.0002
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.0002)
- `--beta1`: Beta1 parameter for Adam optimizer (default: 0.5)
- `--beta2`: Beta2 parameter for Adam optimizer (default: 0.999)
- `--save_interval`: Interval to save model checkpoints (default: 10)

### Evaluation

To evaluate the model and generate metrics like FID and Inception Score:

```bash
python evaluate.py --model_path models/generator.pth
```

Optional arguments:
- `--model_path`: Path to the trained generator model (default: 'models/generator.pth')
- `--num_samples`: Number of samples per class to generate (default: 10)
- `--output_dir`: Directory to save generated images (default: 'generated_images')

## Results

The model's performance is evaluated using:
1. Frechet Inception Distance (FID): Measures the similarity between generated and real image distributions
2. Inception Score (IS): Measures the quality and diversity of generated images
3. Classification accuracy of generated images: Measures if generated images maintain class-specific features

### Sample Results

| Metric | Value |
|--------|-------|
| FID | 68.54 |
| Inception Score | 3.45 ± 0.12 |
| Classification Accuracy | 83.42% |

Detailed results can be found in the project report.

## Project Structure

- `data/`: Contains dataset files and instructions
- `models/`: Contains model architectures
  - `generator.py`: Generator model definition
  - `discriminator.py`: Discriminator model definition
  - `label_embedding.py`: Label embedding network definition
  - `classifier.py`: Classifier model definition
- `utils/`: Contains utility functions
  - `data_loader.py`: Functions for loading and preprocessing data
  - `visualization.py`: Functions for visualizing results
  - `metrics.py`: Functions for computing evaluation metrics
- `train.py`: Main training script
- `evaluate.py`: Evaluation script
- `config.py`: Configuration settings
- `assets/`: Generated images and visualization results

## Training Process

The training process involves:
1. Loading and preprocessing the dataset
2. Training the discriminator to distinguish between real and generated images
3. Training the generator to produce realistic images that fool the discriminator
4. Using the pre-trained classifier to ensure generated images maintain class-specific features
5. Periodic evaluation and saving of model checkpoints

## Visualization

The training progress is visualized using Weights & Biases (wandb), which tracks:
- Discriminator loss (real and fake)
- Generator loss
- Classification loss and accuracy
- Generated image samples
## Acknowledgments

- The ISIC Archive for providing the skin lesion dataset
- The PyTorch team for their excellent deep learning framework
- The Weights & Biases team for their visualization tools
