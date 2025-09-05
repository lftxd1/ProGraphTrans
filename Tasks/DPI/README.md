# DPI Task Introduction

## 1. Task Description

The DPI (Drug-Protein Interaction) task aims to predict the interactions between drug molecules and proteins. This is a crucial step in drug discovery and development, helping to identify potential drug targets and side effects.

## 2. Data Format

The dataset used in this task is the human dataset, which includes the following main components:

- **Drug molecule data**: Represented as molecular graph structures, containing atom features and adjacency matrices
- **Protein data**: Protein sequence features extracted using the ESM-2 pre-trained model
- **Interaction labels**: Binary labels (0 indicates no interaction, 1 indicates interaction)

Data storage paths and formats are defined in the `dataset_config` configuration item in `config.py`.

## 3. Configuration Instructions

Training and model parameters are configured through the `config.py` file, which mainly includes the following sections:

### 3.1 Dataset Configuration
```python
# Dataset configuration
dataset_config = {
    'data_dir': 'dataset/human',  # Dataset root directory
    'feature_dir': 'feature',     # Protein feature directory
    'map_dir': 'map',             # Protein adjacency matrix directory
    'word2vec_dir': 'word2vec_30' # Word vector directory
}
```

### 3.2 Model Configuration
```python
# Model configuration
model_config = {
    'protein_dim': 640,   # Protein feature dimension
    'atom_dim': 34,       # Atom feature dimension
    'hid_dim': 256,       # Hidden layer dimension
    'n_layers': 3,        # Number of network layers
    'n_heads': 8,         # Number of multi-head attention heads
    'pf_dim': 256,        # Feed-forward network dimension
    'kernel_size': 5,     # Convolution kernel size
    'dropout': 0.1,       # Dropout rate
    'use_cuda': True,     # Whether to use CUDA
    'cuda_device': 7      # CUDA device ID
}
```

### 3.3 Training Configuration
```python
# Training configuration
train_config = {
    'epochs': 20,               # Number of training epochs
    'batch_size': 64,           # Batch size
    'lr': 0.001,                # Learning rate
    'weight_decay': 1e-4,       # Weight decay
    'decay_interval': 5,        # Learning rate decay interval
    'lr_decay': 0.6,            # Learning rate decay rate
    'random_seed': 1234,        # Random seed
    'k_folds': 5,               # Number of cross-validation folds
    'save_model': True,         # Whether to save the model
    'model_dir': 'model',       # Model save directory
    'log_dir': 'log'            # Log save directory
}
```

## 4. Usage

### 4.1 Environment Preparation

First, ensure that the required dependency packages are installed. Main dependencies include:
- torch==2.1.0
- torch-geometric==2.3.0
- fair-esm==2.0.0
- numpy, pandas, scikit-learn, etc.

### 4.2 Run Training

After ensuring the data and configuration are correct, you can directly run the `train.py` file to start 5-fold cross-validation training:

```bash
python train.py
```

The program will automatically perform 5-fold cross-validation and save the results of each fold and the average results to the configured log directory.

## 5. Model Architecture

The model architecture used in this task mainly includes the following parts:

### 5.1 Protein Feature Extraction (Encoder)
- Use GCN (Graph Convolutional Network) to process protein sequence features
- Combine CNN (Convolutional Neural Network) and self-attention mechanisms to extract multi-scale features

### 5.2 Drug Molecule Feature Extraction (Decoder)
- Use GCN to process drug molecule graph structures
- Combine Transformer layers to model molecular features

### 5.3 Interaction Prediction (Predictor)
- Combine drug and protein features to predict interaction probability
- Use cross-entropy loss function for training

## 6. Result Evaluation

After training is completed, the program will output the following evaluation metrics:

- **Accuracy**: The proportion of correctly classified samples
- **Precision**: The proportion of positive class predictions that are correct
- **Recall**: The proportion of actual positive cases that are correctly predicted
- **F1 Score**: The harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: A metric that comprehensively measures binary classification quality
- **AUC (Area Under ROC Curve)**: The area under the ROC curve, a metric to measure classifier performance

Results will be output to both the console and log files, and saved as CSV files for subsequent analysis.