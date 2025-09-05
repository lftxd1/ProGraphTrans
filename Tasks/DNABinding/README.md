# DNA Binding Protein Prediction Task

This directory contains code for predicting DNA-binding proteins using ProGraphTrans, a novel approach that combines Graph Neural Networks (GNNs) with Transformers to enhance protein representation learning by integrating both sequence and structural features.

## Task Description

DNA-binding proteins are crucial for various biological processes such as DNA replication, transcription, and repair. The goal of this task is to predict whether a given protein sequence can bind to DNA based on its sequence and structural features.

## Data Format

The dataset consists of positive examples (DNA-binding proteins) and negative examples (non-DNA-binding proteins). The data is organized as follows:

- **PDB14189/**: Training dataset
  - `PDB14189_P.txt`: Positive training examples
  - `PDB14189_N.txt`: Negative training examples
  - `feature/`: Precomputed protein embeddings using ESM-2
  - `map/`: Precomputed attention contact maps

- **PDB2272/**: Test dataset
  - `PDB2272_P.txt`: Positive test examples
  - `PDB2272_N.txt`: Negative test examples
  - `feature/`: Precomputed protein embeddings using ESM-2
  - `map/`: Precomputed attention contact maps

## Requirements

- Python 3.7+
- PyTorch 2.1.0
- torch-geometric 2.3.0
- fair-esm 2.0.0
- scikit-learn
- pandas
- numpy

## Configuration

Before running the training script, you can modify the configuration parameters in `config.py` to suit your needs. The configuration file includes:

- **dataset_config**: Paths to the training and test data
- **model_config**: Model architecture parameters
- **train_config**: Training hyperparameters including epochs, batch size, learning rate, etc.

## Usage

1. Ensure all the required dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script with default parameters:
   ```bash
   python train.py
   ```

3. The script will perform 5-fold cross-validation by default and save the results in the `logs/` directory.

## Model Architecture

The model architecture consists of:

1. **GNN Layers**: Graph Attention Networks (GAT) to capture structural information from protein contact maps
2. **CNN Layers**: Convolutional Neural Networks to extract local sequence features
3. **Transformer Layers**: Self-attention mechanisms to capture long-range dependencies in protein sequences

## Results

The training script will output the following evaluation metrics for each fold:

- Accuracy (ACC)
- Precision (Pre)
- Recall/Sensitivity (Sen)
- Specificity (Spe)
- F1 Score (F1)
- Matthews Correlation Coefficient (MCC)
- Area Under the ROC Curve (AUC)

The results will be saved in CSV format in the `logs/` directory and also printed to the console during training.

## Files in This Directory

- `train.py`: Main training script with 5-fold cross-validation
- `config.py`: Configuration file for dataset paths, model parameters, and training settings
- `README.md`: This documentation file
- `logs/`: Directory to store training logs and evaluation results (auto-created during training)
- `models/`: Directory to store trained models (auto-created during training)