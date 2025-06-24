import numpy as np
from typing import List, Dict, Any

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    :param vec1: First vector
    :param vec2: Second vector
    :return: Cosine similarity score
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Avoid division by zero
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)

def filter_and_rank_results(
    answers: List[Dict[Any, Any]], 
    query_embedding: List[float], 
    top_k: int = 3
) -> List[Dict[Any, Any]]:
    """
    Filter and rank search results based on semantic relevance.
    
    :param answers: List of retrieved documents
    :param query_embedding: Embedding of the original query
    :param top_k: Number of top results to return
    :return: Filtered and ranked results
    """
    # Calculate relevance scores
    relevance_scores = []
    for answer in answers:
        # Ensure 'embedding' exists in the document
        doc_embedding = answer.get('embedding', [])
        
        # Calculate semantic similarity
        if doc_embedding:
            similarity = calculate_cosine_similarity(query_embedding, doc_embedding)
            relevance_scores.append((answer, similarity))
        else:
            relevance_scores.append((answer, 0.0))
    
    # Sort by relevance score (descending)
    sorted_results = sorted(relevance_scores, key=lambda x: x[1], reverse=True)
    
    # Return top K results
    return [result[0] for result in sorted_results[:top_k]]

