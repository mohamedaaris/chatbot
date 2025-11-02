import os
import pickle
from typing import List, Optional, Tuple
import numpy as np

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """Utility to manage a simple FAISS-like vector store with Ollama embeddings.

    This class encapsulates chunking, embedding, and persistence. It is designed
    so it can be reused across multiple Flask routes and future background tasks.
    """

    def __init__(self, persist_directory: str = "vector_store", model: str = "llama3.2:3b") -> None:
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Using Ollama embeddings running locally; ensure Ollama is running and model is pulled
        self.embeddings = OllamaEmbeddings(model=model)
        
        # Simple in-memory storage for documents and embeddings
        self.documents: List[str] = []
        self.doc_embeddings: List[List[float]] = []
        self.metadatas: List[dict] = []
        
        # Default text splitter for long documents
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        
        # Load existing data if available
        self._load_data()

    def _save_data(self) -> None:
        """Save the vector store data to disk."""
        data = {
            'documents': self.documents,
            'embeddings': self.doc_embeddings,
            'metadatas': self.metadatas
        }
        with open(os.path.join(self.persist_directory, 'vector_store.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def _load_data(self) -> None:
        """Load the vector store data from disk."""
        filepath = os.path.join(self.persist_directory, 'vector_store.pkl')
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.doc_embeddings = data.get('embeddings', [])
                    self.metadatas = data.get('metadatas', [])
            except Exception as e:
                print(f"Warning: Could not load vector store data: {e}")

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        return dot_product / (magnitude_a * magnitude_b)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Split, embed, and persist texts into the vector store."""
        # Split texts into chunks for better retrieval
        chunks = []
        metas = []
        for idx, text in enumerate(texts):
            splits = self.splitter.split_text(text)
            chunks.extend(splits)
            base_meta = (metadatas[idx] if metadatas and idx < len(metadatas) else {})
            metas.extend([base_meta] * len(splits))

        if not chunks:
            return

        # Generate embeddings for new chunks
        try:
            new_embeddings = self.embeddings.embed_documents(chunks)
            
            # Add to storage
            self.documents.extend(chunks)
            self.doc_embeddings.extend(new_embeddings)
            self.metadatas.extend(metas)
            
            # Save to disk
            self._save_data()
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[dict]:
        """Perform a similarity search and return documents."""
        if not self.documents:
            return []
            
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.doc_embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity and get top k
            similarities.sort(reverse=True)
            top_k_indices = [i for _, i in similarities[:k]]
            
            # Return documents with metadata
            results = []
            for idx in top_k_indices:
                results.append({
                    'page_content': self.documents[idx],
                    'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {}
                })
            
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []


