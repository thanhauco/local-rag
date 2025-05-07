"""
Local RAG System - Generator Module

Provides LLM generation using HuggingFace's FLAN-T5-Base model
via the transformers pipeline for instruction-following tasks.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    Pipeline,
)
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models import BaseLLM

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Container for generation results."""
    prompt: str
    response: str
    model: str
    tokens_generated: int
    
    def __str__(self) -> str:
        return self.response


class GenerationError(Exception):
    """Exception raised when generation fails."""
    pass


class LLMGenerator:
    """
    LLM Generator using HuggingFace FLAN-T5-Base model.
    
    FLAN-T5 is an instruction-tuned model that excels at:
    - Question answering
    - Summarization
    - Following instructions
    
    Runs locally on CPU without requiring GPU.
    
    Example:
        generator = LLMGenerator()
        response = generator.generate("Summarize this document...")
        print(response)
    """
    
    _instance: Optional['LLMGenerator'] = None
    _pipeline: Optional[Pipeline] = None
    _llm: Optional[HuggingFacePipeline] = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum generation length
            temperature: Sampling temperature
        """
        self.model_name = model_name or config.llm.model_name
        self.max_length = max_length or config.llm.max_length
        self.temperature = temperature or config.llm.temperature
        self.device = config.llm.device
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self._pipeline is None:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        """Initialize the HuggingFace pipeline."""
        try:
            self.logger.info(f"Loading LLM: {self.model_name}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline
            self._pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                device=-1 if self.device == "cpu" else 0,
            )
            
            self.logger.info(f"LLM loaded successfully: {self.model_name}")
            
        except Exception as e:
            raise GenerationError(
                f"Failed to initialize LLM: {str(e)}"
            ) from e
    
    @property
    def pipeline(self) -> Pipeline:
        """Get the HuggingFace pipeline."""
        if self._pipeline is None:
            self._initialize_pipeline()
        return self._pipeline
    
    def get_langchain_llm(self) -> HuggingFacePipeline:
        """
        Get a LangChain-compatible LLM wrapper.
        
        Returns:
            HuggingFacePipeline instance
        """
        if self._llm is None:
            self._llm = HuggingFacePipeline(pipeline=self.pipeline)
            self.logger.info("Created LangChain LLM wrapper")
        
        return self._llm
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Override max length
            temperature: Override temperature
            
        Returns:
            GenerationResult with response
        """
        if not prompt or not prompt.strip():
            raise GenerationError("Cannot generate from empty prompt")
        
        try:
            self.logger.debug(f"Generating from prompt: {prompt[:50]}...")
            
            # Use overrides if provided
            generation_kwargs = {}
            if max_length:
                generation_kwargs["max_length"] = max_length
            if temperature:
                generation_kwargs["temperature"] = temperature
            
            # Generate response
            outputs = self.pipeline(prompt, **generation_kwargs)
            
            response = outputs[0]["generated_text"]
            
            result = GenerationResult(
                prompt=prompt,
                response=response,
                model=self.model_name,
                tokens_generated=len(response.split()),
            )
            
            self.logger.info(
                f"Generated {result.tokens_generated} tokens"
            )
            
            return result
            
        except Exception as e:
            raise GenerationError(
                f"Failed to generate: {str(e)}"
            ) from e
    
    def generate_batch(
        self,
        prompts: List[str],
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of GenerationResult objects
        """
        results = []
        
        for prompt in prompts:
            try:
                result = self.generate(prompt)
                results.append(result)
            except GenerationError as e:
                self.logger.error(f"Failed to generate: {e}")
                results.append(GenerationResult(
                    prompt=prompt,
                    response=f"Error: {str(e)}",
                    model=self.model_name,
                    tokens_generated=0,
                ))
        
        return results
    
    def answer_question(
        self,
        question: str,
        context: str,
    ) -> GenerationResult:
        """
        Answer a question given context.
        
        Args:
            question: The question to answer
            context: Context containing the answer
            
        Returns:
            GenerationResult with answer
        """
        prompt = self._format_qa_prompt(question, context)
        return self.generate(prompt)
    
    def _format_qa_prompt(self, question: str, context: str) -> str:
        """Format a QA prompt for FLAN-T5."""
        return f"""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
    
    def summarize(self, text: str) -> GenerationResult:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            
        Returns:
            GenerationResult with summary
        """
        prompt = f"Summarize the following text:\n\n{text}"
        return self.generate(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "device": self.device,
        }


def get_llm() -> HuggingFacePipeline:
    """
    Get the LangChain LLM instance.
    
    Returns:
        Configured HuggingFacePipeline
    """
    return LLMGenerator().get_langchain_llm()


def generate(prompt: str) -> str:
    """
    Convenience function for text generation.
    
    Args:
        prompt: Input prompt
        
    Returns:
        Generated text
    """
    result = LLMGenerator().generate(prompt)
    return result.response


def answer_question(question: str, context: str) -> str:
    """
    Convenience function for question answering.
    
    Args:
        question: Question to answer
        context: Context containing answer
        
    Returns:
        Answer text
    """
    result = LLMGenerator().answer_question(question, context)
    return result.response
