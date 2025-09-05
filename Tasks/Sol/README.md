# Sol Task Description

## Task Description

The Sol task (Protein Water Solubility Prediction) aims to predict the water solubility of proteins, which is an important property for protein function and stability. This task uses deep learning methods combined with protein sequence features and structural information for prediction.

## Data Format

### Input Data

1. **Training and test data files**: Stored in CSV format, containing protein names, solubility labels, and amino acid sequences
2. **Feature files**: Protein features extracted using ESM-2 model, stored as numpy arrays
3. **Contact map files**: Protein contact map information, stored as numpy arrays

### Data Directory Structure

```
proteinwater/
  ├── eSol_train.csv  # Training data
  ├── eSol_test.csv   # Test data
  ├── feature/        # Protein feature directory
  └── map/            # Protein contact map directory
```

## Configuration Instructions

All configuration parameters are defined in the `config.py` file, including the following main sections:

### Dataset Configuration
```python
dataset_config = {
    'dir_name': 'proteinwater',   # Data directory name
    'train_file': 'eSol_train.csv', # Training data file
    'test_file': 'eSol_test.csv',   # Test data file
    'feature_dir': 'feature',       # Feature directory
    'map_dir': 'map',               # Contact map directory
}
```

### Model Configuration
```python
model_config = {
    'num_node_features': 1280,      # Node feature dimension (ESM-2 output dimension)
    'hidden_feature': 128,          # Hidden layer feature dimension
    'use_cuda': True,               # Whether to use CUDA
    'cuda_device': 7,               # CUDA device number (consistent with original code)
}
```

### Training Configuration
```python
train_config = {
    'k_folds': 5,                   # 5-fold cross-validation
    'epochs': 5,                    # Number of training epochs (consistent with original code)
    'batch_size': 16,               # Batch size
    'lr': 0.0004,                   # Learning rate
    'weight_decay': 0.03,           # Weight decay
    'random_seed': 1234,            # Random seed (consistent with original code)
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

Execute the following command in the `Sol` directory:
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

1. **Graph Convolutional Network (GCN)**: Utilizes graph convolutional network to extract structural features of proteins
   - Uses GATConv layer for graph convolution operations
   - Combines multi-scale CNN to extract sequence features
   - Uses Transformer attention mechanism to enhance feature representation

2. **Attention Mechanism**: Uses Scaled Dot-Product Attention and Multi-Head Attention to capture long-range dependencies in protein sequences
   - Uses protein contact maps as part of attention weights
   - Extracts features from different perspectives through multi-head attention mechanism

3. **Feature Fusion and Prediction**: Fuses the extracted features and performs final prediction through fully connected layers
   - Uses sigmoid activation function to map outputs to [0, 1] interval

## Result Evaluation

The evaluation metrics of the model on the test set include:

- **RMSE (Root Mean Square Error)**: Measures the average deviation between predicted and true values
- **R² (Coefficient of Determination)**: Measures the model's ability to explain data variation
- **Accuracy**: The proportion of correctly predicted samples
- **Precision**: The proportion of positive examples that are correctly predicted
- **Recall**: The proportion of actual positive examples that are correctly predicted
- **F1 Score**: The harmonic mean of precision and recall
- **AUC (Area Under ROC Curve)**: The area under the ROC curve, which measures the model's discrimination ability

5-fold cross-validation calculates the mean and standard deviation of these metrics to comprehensively evaluate model performance.