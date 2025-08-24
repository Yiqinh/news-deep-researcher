import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List

class Searcher:
    """
    A class to search for relevant documents in a FAISS vector store.
    """
    def __init__(self, index_name: str, model_name: str):
        """
        Initializes the Searcher by loading the embedding model and the FAISS index.

        Args:
            index_name (str): The path to the saved FAISS index directory.
            model_name (str): The name of the Hugging Face model used for embeddings.
                               This must be the same model used to create the index.
        """
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        print(f"Loading FAISS index from: {index_name}")
        self.vectorstore = FAISS.load_local(
            index_name, 
            self.embeddings,
            allow_dangerous_deserialization=True 
        )

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search against the loaded vector store.

        Args:
            query (str): The search query string.
            k (int): The number of top documents to retrieve. Defaults to 4.

        Returns:
            List[Document]: A list of the most relevant Document objects.
        """
        print(f"Searching for top {k} results for query: '{query}'")
        results = self.vectorstore.similarity_search(query, k=k)
        return results