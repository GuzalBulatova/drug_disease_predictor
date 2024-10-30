"""Module with Node2Vec Embedder."""

from abc import ABC, abstractmethod
from load_graph import BioKnowledgeGraph
import numpy as np
from node2vec import Node2Vec
from typing import Dict, Optional
import pickle

class GraphEmbedder(ABC):
    """Base class for graph embedding methods."""
    
    def __init__(self, knowledge_graph: BioKnowledgeGraph):
        self.kg = knowledge_graph
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        
    @abstractmethod
    def generate_embeddings(self, dimensions: int = 128) -> Dict[str, np.ndarray]:
        """Generate embeddings for all nodes in the graph"""
        pass
    
    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific node"""
        if self.embeddings is None:
            raise ValueError("Embeddings have not been generated yet")
        return self.embeddings.get(node_id)


class Node2VecEmbedder(GraphEmbedder):
    """Node2Vec implementation of graph embeddings"""
    
    def __init__(self, knowledge_graph: BioKnowledgeGraph):
        super().__init__(knowledge_graph)
    
    def generate_embeddings(self,
                          dimensions: int = 128,
                          walk_length: int = 30,
                          num_walks: int = 200,
                          p: float = 1.0,
                          q: float = 1.0) -> Dict[str, np.ndarray]:
        """Generate Node2Vec embeddings.
        
        Parameters:
        -----------
        dimensions: int, default=128
            Dimensionality of embeddings
        walk_length: int, default=30
            Length of each random walk
        num_walks: int, default=200
            Number of random walks per node
        p: float, default=1.0
            Return parameter (controls likelihood of returning to previous node)
            Default is 1.0, equally likely to backtrack, so walks are balancing
            between staying in local neighbourhood and exploring the new areas
        q: float, deafult=1.0
            In-out parameter (controls search behavior: DFS vs BFS)
            Default=1.0, which balances depth-first-like and breadth-first-like 
            exploration, i.e. regular random walks.
        """
        print(f"Generating {dimensions}-dimensional Node2Vec embeddings...")
        
        # Initialize and train node2vec model
        node2vec = Node2Vec(
            graph=self.kg.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=4  # Parallel processing
        )
        
        model = node2vec.fit(window=10, min_count=1)
        
        # Store embeddings
        self.embeddings = {
            str(node): model.wv[str(node)] 
            for node in self.kg.graph.nodes()
        }
        
        print(f"Generated embeddings for {len(self.embeddings)} nodes")
        return self.embeddings


def save_embeddings(embedder: Node2VecEmbedder, filename: str) -> None:
    """
    Save embeddings to a pickle file
    
    Parameters:
    -----------
    embedder: Node2VecEmbedder
        Trained embedder with generated embeddings
    filename: str
        Path to save the embeddings
    """
    if embedder.embeddings is None:
        raise ValueError("No embeddings have been generated yet")
        
    # Save embeddings dictionary
    with open(filename, 'wb') as f:
        pickle.dump(embedder.embeddings, f)
        
    print(f"Saved embeddings to {filename}")


def load_embeddings(embedder: Node2VecEmbedder, filename: str) -> None:
    """
    Load embeddings from a pickle file
    
    Parameters:
    -----------
    embedder: Node2VecEmbedder
        Embedder instance to load embeddings into
    filename: str
        Path to the saved embeddings
    """
    # Load embeddings dictionary
    with open(filename, 'rb') as f:
        embedder.embeddings = pickle.load(f)

    print(f"Loaded embeddings for {len(embedder.embeddings)} nodes")
