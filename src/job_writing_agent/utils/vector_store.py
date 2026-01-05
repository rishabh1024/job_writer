"""
Vector storage utilities for the job writer application.

This module provides functions for storing and retrieving
documents from vector databases.
"""

# Standard library imports
import os

# Third-party library imports
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Default configuration
DEFAULT_PINECONE_INDEX = "job-writer-vector"


class VectorStoreManager:
    """Manager class for vector store operations."""

    def __init__(
        self,
        index_name: str = DEFAULT_PINECONE_INDEX,
        embedding_model: str = "llama3.2:latest",
    ):
        """Initialize the vector store manager.

        Args:
            api_key: Pinecone API key (will use env var if not provided)
            index_name: Name of the Pinecone index to use
            embedding_model: Name of the Ollama model to use for embeddings
        """
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable PINECONE_API_KEY not set.")

        self.index_name = index_name

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Initialize Pinecone client
        self.client = PineconeClient(api_key=api_key)

        # Ensure index exists
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Make sure the required index exists, create if not."""
        # Get embedding dimension from our embeddings model
        try:
            sample_embedding = self.embeddings.embed_query("Test query")
            embedding_dim = len(sample_embedding)
        except Exception as e:
            print(f"Error determining embedding dimension: {e}")
            print("Falling back to default dimension of 384")
            embedding_dim = 384  # Common default for Ollama embeddings

        # Check if the index exists
        index_exists = False
        try:
            index_list = self.client.list_indexes()
            index_list = [i.name for i in index_list]
            index_exists = self.index_name in index_list
        except Exception as e:
            print(f"Error checking Pinecone indexes: {e}")

        # Create index if it doesn't exist
        if not index_exists:
            try:
                print(f"Creating new index: {self.index_name}")
                self.client.create_index(
                    name=self.index_name,
                    dimension=embedding_dim,
                    spec=ServerlessSpec(region="us-east-1", cloud="aws"),
                    metric="cosine",
                )
                print(f"Successfully created index: {self.index_name}")
            except Exception as e:
                if "ALREADY_EXISTS" in str(e):
                    print(
                        f"Index {self.index_name} already exists (created in another process)"
                    )
                else:
                    print(f"Error creating index: {e}")
        else:
            print(f"Using Pinecone Index: {self.index_name}")

    def store_documents(self, docs: list[Document], namespace: str) -> None:
        """Store documents in vector database.

        Args:
            docs: List of Document objects to store
            namespace: Namespace to store documents under
        """
        try:
            # Get the index
            index = self.client.Index(self.index_name)

            # Create the vector store
            vector_store = Pinecone(
                index=index,
                embedding=self.embeddings,
                text_key="text",
                namespace=namespace,
            )

            # Add documents
            vector_store.add_documents(docs)
            print(
                f"Successfully stored {len(docs)} documents in namespace: {namespace}"
            )
        except Exception as e:
            print(f"Error storing documents: {e}")
            raise

    def retrieve_similar(
        self, query: str, namespace: str, k: int = 3
    ) -> list[Document]:
        """Retrieve similar documents based on a query.

        Args:
            query: The query text to search for
            namespace: Namespace to search in
            k: Number of results to return

        Returns:
            List of Document objects
        """
        try:
            # Get the index
            index = self.client.Index(self.index_name)

            # Create the vector store
            vectorstore = Pinecone(
                index=index,
                embedding=self.embeddings,
                text_key="text",
                namespace=namespace,
            )

            # Search for similar documents
            docs = vectorstore.similarity_search(query, k=k, namespace=namespace)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []


# Example usage (commented out to prevent auto-execution)
# vector_store_manager = VectorStoreManager()
# vector_store_manager.store_documents(
#     docs=[Document(page_content="Sample content", metadata={"source": "test"})],
#     namespace="test_namespace"
# )
