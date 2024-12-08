
# README: Advanced Persistent Threat Detection Framework

## Project Overview
This project implements a framework for detecting Advanced Persistent Threats (APTs) using Masked Graph Representation Learning. The solution uses various datasets, including CADETS, THEIA, and TRACE, and supports data preprocessing, training, and evaluation using state-of-the-art machine learning techniques. 

---

## Code Modules

### **1. Data Parsing**
- **`trace_parser.py`**: Parses TRACE, THEIA, and CADETS datasets.
- **`wget_parser.py`**: Parses WGET dataset.
- **`streamspot_parser.py`**: Parses StreamSpot dataset.

### **2. Model Components**
- **`autoencoder.py`**: Implements the autoencoder architecture for anomaly detection.
- **`gat.py`**: Contains the Graph Attention Network (GAT) model for graph-based learning.
- **`mlp.py`**: Defines a Multi-Layer Perceptron (MLP) used for classification tasks.

### **3. Core Workflow**
- **`train.py`**: Trains the models using the parsed datasets and selected architectures.
- **`eval.py`**: Evaluates model performance on the datasets and generates results.

---

## Datasets


- **StreamSpot Dataset:** This simulated dataset, collected using SystemTap, contains 600 batches of audit logs covering six scenarios—five benign user behaviors and one simulating a drive-by-download attack. It is relatively small and lacks labels for log entries or system entities, requiring batched log-level detection similar to prior works.

- **Unicorn Wget Dataset:** Designed by Unicorn, this dataset includes 150 batches of logs collected with Camflow, where 125 are benign and 25 involve stealthy supply-chain attacks resembling benign workflows. It is challenging due to its large volume, complex log format, and the stealth nature of attacks, with detection performed at the batched log level.

- **DARPA Engagement 3 Datasets:** Part of the DARPA Transparent Computing program, this dataset involves APT attacks on an enterprise network, with blue teams auditing and analyzing system causality. It includes the Trace, THEIA, and CADETS sub-datasets, totaling 51.69GB of audit logs with millions of system entities and interactions, used for system-entity-level detection and overhead analysis.
---

## Code Files In Details

![image](https://github.com/user-attachments/assets/91b1e312-a6a4-4bee-a9b5-34814a477267)


### **checkpoints Folder**
In this folder, trained models are saved. After training the model on a specific dataset, it is saved there.

### **data Folder**
This folder contains five sub-folders. Each one is considered a container of a certain dataset. It has the raw data, and when parsers are run, it contains the preprocessed datasets.

#### DARPA Datasets 3 Folders:

- *Raw Files:* .pkl files that are parsed to TEXT files and JSON files.

- *JSON Files:* an intermediate stage between the zipped files and TEXT file in which it has the data structured to represent the number of nodes feature dimensions, number of edge vector dimensions and the IDs of the malicious source and distination nodes, and the number of training files, number of test files. 

- *TEXT Files:* the Final form of the files which model is trained on. It has each source and destination nodes type, source node ID, target node ID, 

#### StreamSpot Dataset Folder:

- *Raw Files:* 600 .tsv files that are converted to JSON files.

- *JSON Files:* Each JSON file corresponds to a graph, and each graph consists of IDs of the nodes and the type of each node, they are represented in numbers and letters. 

#### UNICORN Wget Dataset Folder:

- *Raw Files:* .tsv files that are converted to log files.

- *Log Files:* Each log file contains the IDs of source and destination nodes IDs, and the type of process happened between them. 

### **model Folder**
It consists of the following files:
- autoencoder.py: it has the code that implements a Graph Masked Autoencoder (GMAE), designed for graph-structured data. The GMAE is built on PyTorch and DGL, leveraging Graph Attention Networks (GAT) for encoding and decoding graph features. It supports feature masking, attribute reconstruction, and structural reconstruction for graph-based machine learning tasks.

- eval.py: It contains functions for evaluating machine learning models at both batch and entity levels. Here's an overview of the modifications and functionalities included: (A) Batch-Level Evaluation: It evaluates models based on graph-level embeddings using k-NN computing AUC, precision, recall, F1-score, and confusion matrix metrics.
(B) Entity-Level Evaluation: It evaluates models based on individual entity embeddings (e.g., nodes or records) using k-NN nased on thresholds for each dataset like trace, theia, and cadets.

- gat.py: This implementation provides a customizable Graph Attention Network (GAT) model using PyTorch and DGL. (A) GAT Module: A multi-layer GAT architecture, allowing for flexible configurations such as the number of layers, attention heads, hidden dimensions, and activation functions. Features dropout, normalization, residual connections, and optional concatenation of attention head outputs. (B) GATConv Module: Implements a single GAT layer with edge and node attention mechanisms. It includes multi-head attention, edge feature transformations, and learnable parameters for attention scores. It leverages DGL's message-passing API to compute attention-weighted outputs. 

- train.py: It provides a batch-level training routine for Graph Neural Network (GNN) models using the DGL library. It handles the transformation, batching, and optimization of graph data during the training process. 

- loss_func.py / mlp.py: They define two essential components for tasks like structure reconstruction in Graph Neural Networks (GNNs): a simple Multi-Layer Perceptron (MLP) and a Soft Cross Entropy (SCE) Loss Function. These components are lightweight, modular, and highly reusable in different GNN-based workflows.

### **utils Folder**
It consists of the following files:
- config.py: It just parses the input command line arguments.

- loaddata.py: It provides functionality for loading, processing, and managing datasets for graph-based learning, specifically tailored for node and edge classification. It integrates frameworks like DGL, NetworkX, and PyTorch for efficient handling of graph data.

- poolers.py: It defines a custom pooling layer named Pooling using PyTorch, designed for aggregating features from graph nodes. The pooling operation can apply to the entire graph or be conditioned on node types, allowing for type-specific feature aggregation.

- streamspot_parser.py: It processes the datasets of graph information, stored in a tab-separated values (TSV) file, and splits it into individual graph files in JSON format, which can be used for further analysis or training machine learning models. The code uses the networkx library to work with directed graphs and applies filters based on node and edge types.

- wget_parser.py: It processes log files to extract graph data, converting raw log entries into directed graphs (DiGraphs) that are serialized into JSON format. It works by reading log files, parsing the edges, and filtering them based on node and edge types. 

- trace_parser.py: processes graph data from a specified dataset. The code processes each file containing graph data, where each line represents an edge between two nodes. Each edge has associated attributes: source node (src), destination node (dst), their types (src_type, dst_type), the edge type (edge_type), and a timestamp (ts).

- utils.py: It provides utility functions and modules to support neural network training and optimization, with a focus on flexibility and modularity for tasks such as selecting optimizers, activations, normalization layers, and managing random seeds.
---

## Run

This is a guildline on reproducing MAGIC's evaluations. There are three options: **Quick Evaluation**, **Standard Evaluation** and **Training from Scratch**.

### Quick Evaluation

Make sure you have MAGIC's parameters saved in `checkpoints/` and KNN distances saved in `eval_result/`. Then execute `eval.py` and assign the evaluation dataset using the following command:
```
  python eval.py --dataset *your_dataset*
```
### Standard Evaluation

Standard evaluation trains the detection module from scratch, so the KNN distances saved in `eval_result/` need to be removed. MAGIC's parameters in `checkpoints/` are still needed. Execute `eval.py` with the same command to run standard evaluation:
```
  python eval.py --dataset *your_dataset*
```
### Training from Scratch

Namely, everything, including MAGIC's graph representation module and its detection module, are going to be trained from raw data. Remove model parameters from `checkpoints/` and saved KNN distances from `eval_result/` and execute `train.py` to train the graph representation module. 
```
  python train.py --dataset *your_dataset*
```
Then execute `eval.py` the same as in standard evaluation:
```
  python eval.py --dataset *your_dataset*
```
For more running options, please refer to `utils/config.py`

- **Authors** Zian Jia, Yun Xiong, Yuhong Nan, Yao Zhang, Jinjing Zhao, Mi Wen  
- **Students:** Cenker Sengoz, Devon Blewett, and Magdy Nasr  
- **Course:** COMP 7860  
- **Instructor:** Prof. Azadeh Tabiban  
- **Date:** December 9, 2024  


## References
1. Zian Jia, et al. "MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning." USENIX Security 2024. [Paper Link](https://www.usenix.org/conference/usenixsecurity24/presentation/jia-zian)
2. DARPA Transparent Computing Dataset: [GitHub Link](https://github.com/darpa-i2o/Transparent-Computing).  
3. StreamSpot Dataset: [GitHub Link](https://github.com/sbustreamspot/sbustreamspot-data).  
4. Unicorn Dataset: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IA8UOS).
5. MAGIC Github Repository: [GitHub Link](https://github.com/FDUDSDE/MAGIC)
---
