# PPI Task Description

## Task Description

Protein-Protein Interaction (PPI) prediction is an important problem in bioinformatics. This task aims to predict whether an interaction exists between two proteins.

## Data Format

### Input Data

1. **Protein dictionary file**: TSV file storing protein names and sequence information
2. **Feature files**: Protein features extracted using ESM-2 model, stored as numpy arrays
3. **Contact map files**: Protein contact map information, stored as numpy arrays
4. **Training and test data files**: TSV files containing protein pairs and interaction labels

### Data Directory Structure

```
PP/
  ├── protein.dictionary.tsv  # Protein dictionary file
  ├── feature/                # Protein feature directory
  ├── map/                    # Protein contact map directory
  ├── train_cmap.actions.tsv  # Training data
  └── test_cmap.actions.tsv   # Test data
```

## Configuration Instructions

All configuration parameters are defined in the `config.py` file, including the following main sections:

### Dataset Configuration
```python
dataset_config = {
    'protein_dict_file': 'PP/protein.dictionary.tsv',  # Protein dictionary file path
    'feature_dir': 'PP/feature',                       # Protein feature directory
    'map_dir': 'PP/map',                               # Protein contact map directory
    'train_file': 'PP/train_cmap.actions.tsv',         # Training data file
    'test_file': 'PP/test_cmap.actions.tsv',           # Test data file
}
```

### Model Configuration
```python
model_config = {
    'num_node_features': 1280,      # Node feature dimension (ESM-2 output dimension)
    'hidden_feature': 256,          # Hidden layer feature dimension
    'use_cuda': True,               # Whether to use CUDA
    'cuda_device': 0,               # CUDA device number
}
```

### Training Configuration
```python
train_config = {
    'k_folds': 5,                   # 5-fold cross-validation
    'epochs': 40,                   # Number of training epochs
    'batch_size': 8,                # Batch size
    'lr': 0.0005,                   # Learning rate
    'weight_decay': 0,              # Weight decay
    'random_seed': 42,              # Random seed
    'save_model': True,             # Whether to save the model
    'model_dir': 'models',          # Model save directory
    'log_dir': 'logs',              # Log save directory
}
```

## Usage

### 1. Install Dependencies

Ensure the following dependency packages are installed:
```
pytorch == 2.1.0
pytorch-geometric == 2.3.0
fair-esm == 2.0.0
pandas
numpy
scikit-learn
```

### 2. Prepare Data

Ensure the data directory structure and files meet the above requirements.

### 3. Run Training

Execute the following command in the `PPI` directory:
```bash
python train.py
```

### 4. View Results

- Training logs will be displayed in the console and saved to the `logs/training.log` file
- Detailed results of 5-fold cross-validation will be saved to the `logs/cross_validation_results.csv` file
- Average results will be saved to the `logs/average_results.csv` file
- The best model for each fold will be saved to the `models` directory

## Model Architecture

The model used in this task consists of the following main components:

1. **GCN Network**: Utilizes Graph Convolutional Network to extract structural features of proteins
   - Uses GATConv layer for graph convolution operations
   - Combines CNN feature extraction
   - Uses Transformer attention mechanism to enhance feature representation

2. **PPI Main Model**: Combines two GCN networks to process protein pairs
   - Extracts feature representations of two proteins separately
   - Fuses features through fully connected layers and performs binary classification

3. **Attention Mechanism**: Uses Scaled Dot-Product Attention and Multi-Head Attention to enhance feature learning

## Result Evaluation

The evaluation metrics of the model on the test set include:

- **Accuracy**: The proportion of correctly predicted samples
- **Precision**: The proportion of positive examples that are correctly predicted
- **Recall/Sensitivity**: The proportion of actual positive examples that are correctly predicted
- **Specificity**: The proportion of actual negative examples that are correctly predicted
- **F1 Score**: The harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: A metric that comprehensively measures classification quality
- **AUC (Area Under ROC Curve)**: The area under the ROC curve, which measures the model's discrimination ability

5-fold cross-validation calculates the mean and standard deviation of these metrics to comprehensively evaluate model performance.