"""Module with KG class."""

import pandas as pd
import networkx as nx
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

class BioKnowledgeGraph:
    def __init__(self, nodes_file: str, edges_file: str) -> None:
        """
        Initialize the knowledge graph from CSV files containing nodes and edges
        
        Parameters:
        -----------
        nodes_file: str
            Path to the CSV file containing nodes
        edges_file: str
            Path to the CSV file containing edges
        """
        # Load data
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        
        # Create NetworkX graph
        self.graph = nx.MultiDiGraph()
        
        # Load nodes
        for _, row in self.nodes_df.iterrows():
            # Convert all names and categories to lists
            all_names = row['all_names'].split('ǂ') if isinstance(row['all_names'], str) else []
            all_categories = row['all_categories'].split('ǂ') if isinstance(row['all_categories'], str) else []
            
            # Add node with all attributes
            self.graph.add_node(
                row['id'],
                name=row['name'],
                category=row['category'],
                all_names=all_names,
                all_categories=all_categories,
                description=row['description'],
                iri=row['iri']
            )
        
        # Load edges
        for _, row in self.edges_df.iterrows():
            # Add edge with all attributes
            self.graph.add_edge(
                row['subject'],
                row['object'],
                predicate=row['predicate'],
                knowledge_source=row['knowledge_source'],
                publications=row['publications'],
                type=row['type']
            )
            
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def find_edges(self,
                subject: Optional[str] = None,
                predicate: Optional[str] = None,
                object: Optional[str] = None,
                object_category: Optional[str] = None) -> List[Tuple[str, str, Dict]]:
        """
        Find edges matching the specified criteria
        
        Parameters:
        -----------
        subject: Optional[str]
            Subject (source) ID to filter by
        predicate: Optional[str]
            Predicate type to filter by
        object: Optional[str]
            Object (target) ID to filter by
        object_category: Optional[str]
            Category of object node to filter by
            
        Returns:
        --------
        List[Tuple[str, str, Dict]]
            List of (subject, object, attributes) tuples
        """
        results = []
        
        for u, v, data in self.graph.edges(data=True):
            # Check subject (source node)
            if subject and u != subject:
                continue
                
            # Check object (target node)
            if object and v != object:
                continue
                
            # Check predicate
            if predicate and data.get('predicate') != predicate:
                continue
            
            # Check object category
            if object_category:
                target_node = self.graph.nodes[v]
                # Check primary category
                if target_node.get('category') != object_category:
                    # If not in primary category, check all_categories
                    all_cats = target_node.get('all_categories', [])
                    if isinstance(all_cats, str):
                        all_cats = all_cats.split('ǂ')
                    if object_category not in all_cats:
                        continue
            
            results.append((u, v, data))
            
        return results
    
    def get_edge(self, subject: str, object: str) -> Optional[Dict]:
        """
        Get the attributes of an edge between two nodes
        
        Parameters:
        -----------
        subject: str
            Subject node ID
        object: str
            Object node ID
            
        Returns:
        --------
        Optional[Dict]
            Edge attributes if edge exists, None otherwise
        """
        if self.graph.has_edge(subject, object):
            return self.graph.get_edge_data(subject, object)
        return None
    
    def find_paths(self, 
                  start: str, 
                  end: str, 
                  max_length: int = 3) -> List[List[str]]:
        """
        Find all paths between two nodes up to a maximum length
        
        Parameters:
        -----------
        start: str
            Starting node ID
        end: str
            Target node ID
        max_length: int
            Maximum path length
            
        Returns:
        --------
        List[List[str]]
            List of paths (each path is a list of node IDs)
        """
        try:
            return list(nx.all_simple_paths(self.graph, start, end, cutoff=max_length))
        except nx.NetworkXNoPath:
            return []
    
    def has_path(self, start: str, end: str, max_length: int = 3) -> bool:
        """
        Check if there exists a path between two nodes
        
        Parameters:
        -----------
        start: str
            Starting node ID
        end: str
            Target node ID
        max_length: int
            Maximum path length
            
        Returns:
        --------
        bool
            True if a path exists, False otherwise
        """
        return len(self.find_paths(start, end, max_length)) > 0
    
    def find_nodes_by_category(self, category: str) -> List[str]:
        """
        Find all nodes of a specific category
        
        Parameters:
        -----------
        category: str
            Category to search for
            
        Returns:
        --------
        List[str]
            List of node IDs
        """
        return [
            node for node, attrs in self.graph.nodes(data=True)
            if category in attrs.get('all_categories', [])
        ]
    
    def get_node_info(self, node_id: str) -> Optional[Dict]:
        """
        Get all information about a specific node
        
        Parameters:
        -----------
        node_id: str
            Node ID to look up
            
        Returns:
        --------
        Optional[Dict]
            Node attributes if node exists, None otherwise
        """
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

# Examples
if __name__ == "__main__":
    # Load the graph
    graph = BioKnowledgeGraph("data/Nodes.csv", "data/Edges.csv")
    
    # Example how to use
    print("\nExample 1: Finding edges related to a specific gene:")
    gene_edges = graph.find_edges(subject="NCBIGene:8483")
    for source, target, data in gene_edges:
        print(f"{source} --[{data['predicate']}]--> {target}")
    
    print("\nExample 2: Finding all proteins:")
    proteins = graph.find_nodes_by_category("biolink:Protein")
    print(f"Found {len(proteins)} proteins")
    
    print("\nExample 3: Getting information about a specific node:")
    node_info = graph.get_node_info("NCBIGene:8483")
    if node_info:
        print(f"Node name: {node_info['name']}")
        print(f"Node category: {node_info['category']}")
        print(f"Node description: {node_info['description']}")