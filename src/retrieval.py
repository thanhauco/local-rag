"""
Local RAG System - Retrieval Module

Provides retrieval capabilities using LangChain's RetrievalQA chain
for context-aware querying against the vector store.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM

from .config import config
from .vector_store import VectorStoreManager, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    query: str
    answer: str
    source_documents: List[Document]
    retrieval_count: int
    
    @property
    def sources(self) -> List[str]:
        """Get unique source file names."""
        sources = set()
        for doc in self.source_documents:
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
            elif "file_name" in doc.metadata:
                sources.add(doc.metadata["file_name"])
        return list(sources)
    
    def format_answer(self, include_sources: bool = True) -> str:
        """Format the answer with optional sources."""
        result = self.answer
        
        if include_sources and self.sources:
            result += "\n\nSources:\n"
            for source in self.sources:
                result += f"  - {source}\n"
        
        return result


class RetrievalError(Exception):
    """Exception raised when retrieval fails."""
    pass


class RetrievalManager:
    """
    Manager for RAG retrieval operations.
    
    Combines vector store retrieval with LLM generation
    to provide context-aware answers to queries.
    
    Example:
        manager = RetrievalManager(llm=my_llm)
        result = manager.query("What is the main topic?")
        print(result.answer)
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
        top_k: Optional[int] = None,
        return_source_documents: bool = True,
    ):
        """
        Initialize the retrieval manager.
        
        Args:
            llm: Language model for generation
            vector_store_manager: Vector store manager instance
            top_k: Number of documents to retrieve
            return_source_documents: Whether to include source docs
        """
        self.llm = llm
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.top_k = top_k or config.retrieval.top_k
        self.return_source_documents = return_source_documents
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._qa_chain: Optional[RetrievalQA] = None
    
    def get_retriever(self) -> BaseRetriever:
        """
        Get the vector store retriever.
        
        Returns:
            Configured retriever instance
        """
        vector_store = self.vector_store_manager.get_vector_store()
        
        return vector_store.as_retriever(
            search_type=config.retrieval.search_type,
            search_kwargs={
                "k": self.top_k,
            },
        )
    
    def create_qa_chain(self, llm: BaseLLM) -> RetrievalQA:
        """
        Create a RetrievalQA chain.
        
        Args:
            llm: Language model for generation
            
        Returns:
            Configured RetrievalQA chain
        """
        retriever = self.get_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Uses all retrieved docs in context
            retriever=retriever,
            return_source_documents=self.return_source_documents,
            verbose=False,
        )
        
        self.logger.info("Created RetrievalQA chain")
        return qa_chain
    
    def get_qa_chain(self) -> RetrievalQA:
        """
        Get or create the QA chain.
        
        Returns:
            RetrievalQA chain instance
        """
        if self._qa_chain is None:
            if self.llm is None:
                raise RetrievalError(
                    "LLM is required for QA chain. "
                    "Initialize with an LLM or call create_qa_chain()."
                )
            self._qa_chain = self.create_qa_chain(self.llm)
        
        return self._qa_chain
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        retriever = self.get_retriever()
        
        if k and k != self.top_k:
            # Create new retriever with different k
            vector_store = self.vector_store_manager.get_vector_store()
            retriever = vector_store.as_retriever(
                search_type=config.retrieval.search_type,
                search_kwargs={"k": k},
            )
        
        try:
            documents = retriever.invoke(query)
            self.logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            raise RetrievalError(
                f"Failed to retrieve documents: {str(e)}"
            ) from e
    
    def query(
        self,
        question: str,
        llm: Optional[BaseLLM] = None,
    ) -> RetrievalResult:
        """
        Query the RAG system and get an answer.
        
        Args:
            question: The question to ask
            llm: Optional LLM override
            
        Returns:
            RetrievalResult with answer and sources
        """
        if llm:
            qa_chain = self.create_qa_chain(llm)
        else:
            qa_chain = self.get_qa_chain()
        
        try:
            self.logger.info(f"Processing query: {question[:50]}...")
            
            result = qa_chain.invoke({"query": question})
            
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            retrieval_result = RetrievalResult(
                query=question,
                answer=answer,
                source_documents=source_docs,
                retrieval_count=len(source_docs),
            )
            
            self.logger.info(
                f"Generated answer with {len(source_docs)} source documents"
            )
            
            return retrieval_result
            
        except Exception as e:
            raise RetrievalError(
                f"Failed to process query: {str(e)}"
            ) from e
    
    def batch_query(
        self,
        questions: List[str],
        llm: Optional[BaseLLM] = None,
    ) -> List[RetrievalResult]:
        """
        Process multiple queries.
        
        Args:
            questions: List of questions
            llm: Optional LLM override
            
        Returns:
            List of RetrievalResult objects
        """
        results = []
        
        for question in questions:
            try:
                result = self.query(question, llm)
                results.append(result)
            except RetrievalError as e:
                self.logger.error(f"Failed to process: {question}: {e}")
                results.append(RetrievalResult(
                    query=question,
                    answer=f"Error: {str(e)}",
                    source_documents=[],
                    retrieval_count=0,
                ))
        
        return results


def create_retriever(top_k: int = 4) -> BaseRetriever:
    """
    Create a retriever from the vector store.
    
    Args:
        top_k: Number of documents to retrieve
        
    Returns:
        Configured retriever
    """
    manager = RetrievalManager(top_k=top_k)
    return manager.get_retriever()


def retrieve_documents(query: str, k: int = 4) -> List[Document]:
    """
    Convenience function to retrieve documents.
    
    Args:
        query: Search query
        k: Number of documents
        
    Returns:
        List of relevant documents
    """
    manager = RetrievalManager(top_k=k)
    return manager.retrieve_documents(query)
