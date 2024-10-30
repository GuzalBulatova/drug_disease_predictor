# Drug-Disease Association Prediction

A machine learning pipeline that leverages graph embeddings to predict potential drug-disease relationships from biological knowledge graphs.

## To run the code:
1. Clone the repository.
2.1. *Optional: download the embeddings file from shared link `drug_disease_embeddings.pkl`
2. Add the data files for the challenge under data folder (currently empty). Please make sure that files include Nodes, Edges and Ground Truth files. Please insert underscore in the Ground_Truth file name.
3. Install required packages: `pip install -r requirements.txt`
3. To run with
    - default settings (using previously generated embeddings): run `python pipeline.py`
    - generating embeddings: run `python pipeline.py --generate-embeddings.py`
    - predict new drug-disease pairs: run `python pipeline.py --predict-pairs new_pairs.csv`
    NB: in this case please update new_pairs.csv with your drug-disease pairs


## Approach & Design Choices

### 1. Graph Embedding Generation (Node2Vec)
We chose Node2Vec as our embedding method. Our reasoning is the following:
- It captures both structural equivalence and local neighborhoods through flexible random walks
- We assume for biological networks indirect relationships are meaningful, and with Node2Vec we preserve higher-order proximity between nodes
- Used in similar biomedical applications
- It's scalable to large graphs

### 2. Feature Engineering
Our feature vector design combines multiple complementary signals:

#### Embedding-based Features:
- Hadamard product: we aim to capture correlation between corresponding dimensions
- L1 distance (absolute difference): smaller values indicate more similar node pairs
- Average of embeddings: represent the "midpoint" between two nodes in the embedding space
- Element-wise maximum: to capture when either node has a strong feature that might be relevant to the relationship

These capture different aspects of node pair relationships in the embedding space.


### 3. Model Selection (Random Forest)
We selected Random Forest, motivation:
- Handles non-linear relationships in the feature space
- Robust to feature scaling and outliers
- Less prone to overfitting compared to deeper models
- Works well with our heterogeneous feature set

## Model Performance & Analysis
*see results-visualisation.ipynb for plots* 
### Key Metrics
- ROC-AUC scores:
  - Training: ~1
  - Test: ~0.91
- Average Precision scores:
  - Training: ~1
  - Test: ~0.87

### Feature Importance Analysis
Most influential feature types (normalized importance):
1. Element-wise max (0.36)
2. Average (0.24)
4. Hadamard (0.2)
5. L1 distance (0.18)

### Key Observations
1. The model shows strong predictive performance with minimal overfitting (small gap between train and test metrics)
2. Element-wise maximum is the most informative operation for the relationship prediction
3. Model is relying on a broad set of features, not a few predominant ones

### Model Limitations
1. The current feature set might not capture all biological mechanisms underlying drug-disease relationships
2. Binary classification approach might oversimplify the complex nature of drug-disease interactions
3. Performance may be not as good for rare diseases or new drugs with limited graph connectivity

## Future Improvements
1. Further feature engineering:
   - Incorporate skip-gram based features
   - Explore additional graph structural metrics
2. Experiment with knowledge graph embedding methods for example TransE. Other embedding models can explicitly account for different relationship types in the knowledge graph. Node2Vec doesn't do it by default nor in our implementation.
3. Explore multi-task learning to leverage different types of edges in the knowledge graph
