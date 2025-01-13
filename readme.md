# ProGraphTrans: GNN-guided Transformer for Protein Representation Learning

## Introduction

Proteins are essential biomolecules that regulate virtually all biological processes. Determining their function based solely on their sequence remains a significant challenge in bioinformatics. Recent advances have led to the development of models like ESM-2, which learn protein representations to assist in various bioinformatics tasks. However, most current methods focus primarily on sequence information, often neglecting the structural characteristics of proteins.

In this paper, we introduce **ProGraphTrans**, a novel approach that combines Graph Neural Networks (GNNs) with Transformers to enhance protein representation learning by integrating both sequence and structural features. Our method leverages ESM-2 embeddings and amino acid contact profiles to build richer representations, improving the performance of protein-related tasks.

## Key Features of ProGraphTrans

- **Graph Neural Network (GNN)-Guided Attention**: GNNs guide the attention mechanism of a Transformer, allowing the model to focus on key protein residues and capture both sequence and structural features.
- **Multi-Scale Convolutional Neural Networks (CNNs)**: CNNs are used to capture local and global sequence features, ensuring that the model benefits from both fine-grained and global patterns in the protein sequence.
- **ESM-2 Model**: ESM-2 embeddings provide a robust initial representation of proteins, while amino acid contact profiles predict structural information.
- **Versatile and Effective**: ProGraphTrans outperforms existing state-of-the-art methods in tasks like DNA-binding protein prediction, protein-protein interaction prediction, protein solubility prediction, and compound-protein interaction prediction.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- DGL (Deep Graph Library)
- ESM-2 Model (Hugging Face Transformers)
- NumPy, SciPy, and other standard libraries

## Installation

To get started with ProGraphTrans, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/lftxd1/ProGraphTrans.git
cd ProGraphTrans

# Install dependencies
pip install -r requirements.txt
```

## Model Architecture

1. **Protein Sequence Embedding**: Proteins are embedded using the ESM-2 model, which generates high-dimensional embeddings.
2. **Amino Acid Contact Profile**: The predicted contact profile is used to reconstruct the 3D structure of the protein.
3. **Graph Neural Network**: GNNs are used to extract structural features from the protein graph (where nodes represent amino acids and edges represent their interactions).
4. **Convolutional Neural Networks**: These networks capture both local and global sequence information.
5. **Transformer with GNN-guided Attention**: The Transformer network uses GNN-derived attention to focus on important residues, enhancing the model’s ability to understand complex protein functions.

## Tasks and Benchmarks

ProGraphTrans has been validated on multiple protein-related tasks:

1. **DNA-Binding Protein Prediction**: Predict whether a protein binds to DNA based on its sequence and structural features.
2. **Protein-Protein Interaction Prediction**: Predict the interaction between two proteins.
3. **Protein Solubility Prediction**: Predict the solubility of a protein in different conditions.
4. **Compound-Protein Interaction Prediction**: Predict the interaction between a compound and a protein.

The model outperforms state-of-the-art methods across all four tasks, demonstrating its effectiveness in combining sequence and structural data for protein representation learning.

## Contributing

We welcome contributions to improve ProGraphTrans. If you want to add new features, fix bugs, or improve the documentation, feel free to submit a pull request. Please make sure your code follows the existing coding style and passes the tests.

## License

This project is licensed under the MIT License - see the [LICENSE](https://chatgpt.com/c/LICENSE) file for details.