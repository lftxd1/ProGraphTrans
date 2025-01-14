# ProGraphTrans: GNN-guided Transformer for Protein Representation Learning

## Introduction

Proteins are essential biomolecules that regulate virtually all biological processes. Determining their function based solely on their sequence remains a significant challenge in bioinformatics. Recent advances have led to the development of models like ESM-2, which learn protein representations to assist in various bioinformatics tasks. However, most current methods focus primarily on sequence information, often neglecting the structural characteristics of proteins.

In this paper, we introduce **ProGraphTrans**, a novel approach that combines Graph Neural Networks (GNNs) with Transformers to enhance protein representation learning by integrating both sequence and structural features. Our method leverages ESM-2 embeddings and amino acid contact profiles to build richer representations, improving the performance of protein-related tasks.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- [ESM-2](https://github.com/facebookresearch/esm) 
- [transformerCPI](https://github.com/lifanchen-simm/transformerCPI)

## Installation

To get started with ProGraphTrans, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/lftxd1/ProGraphTrans.git
cd ProGraphTrans

# Install dependencies
pip install -r requirements.txt
```

## Tasks and Benchmarks

ProGraphTrans has been validated on multiple protein-related tasks:

1. **DNA-Binding Protein Prediction**: Predict whether a protein binds to DNA based on its sequence and structural features.
2. **Protein-Protein Interaction Prediction**: Predict the interaction between two proteins.
3. **Protein Solubility Prediction**: Predict the solubility of a protein in different conditions.
4. **Compound-Protein Interaction Prediction**: Predict the interaction between a compound and a protein.

## Contributing

We welcome contributions to improve ProGraphTrans. If you want to add new features, fix bugs, or improve the documentation, feel free to submit a pull request. 

## License

This project is licensed under the MIT License - see the [LICENSE]() file for details.