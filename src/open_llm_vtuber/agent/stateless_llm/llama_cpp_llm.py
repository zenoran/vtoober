"""Description: This file contains the implementation of the LLM class using llama.cpp.
This class provides a stateless interface to llama.cpp for language generation.
"""

import asyncio
from typing import AsyncIterator, List, Dict, Any
from llama_cpp import Llama
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface


class LLM(StatelessLLMInterface):
    def __init__(
        self,
        model_path: str,
        **kwargs,
    ):
        """
        Initializes a stateless instance of the LLM class using llama.cpp.

        Parameters:
        - model_path (str): Path to the GGUF model file
        - **kwargs: Additional arguments passed to Llama constructor
        """
        logger.info(f"Initializing llama cpp with model path: {model_path}")
        logger.info(f"Raw kwargs received: {kwargs}")
        logger.info("🚨🚨🚨 NICK - THIS IS THE NEW CODE VERSION - LOOK FOR GPU LAYERS! 🚨🚨🚨")
        
        # Print each kwarg to see what's missing
        for key, value in kwargs.items():
            logger.info(f"  kwargs['{key}'] = {value} (type: {type(value)})")
        
        self.model_path = model_path
        
        # Build parameters exactly like the working llama_cpp_client.py
        model_load_params = {
            "model_path": model_path,
            "n_gpu_layers": kwargs.get("n_gpu_layers", -1),
            "n_ctx": kwargs.get("n_ctx", 4096),
            # "chat_format": "chatml",  # This was missing!
            "verbose": False,  # Use False like working client
        }
        
        # Add optional parameters if present
        if "n_batch" in kwargs:
            model_load_params["n_batch"] = kwargs["n_batch"]
        if "main_gpu" in kwargs:
            model_load_params["main_gpu"] = kwargs["main_gpu"]
        if "tensor_split" in kwargs and kwargs["tensor_split"] is not None:
            model_load_params["tensor_split"] = kwargs["tensor_split"]
        
        # Log what we're actually passing to llama-cpp-python
        logger.info(f"llama-cpp-python parameters: {model_load_params}")
        
        try:
            self.llm = Llama(**model_load_params)
        except Exception as e:
            logger.critical(f"Failed to initialize Llama model: {e}")
            raise

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str = None
    ) -> AsyncIterator[str]:
        """
        Generates a chat completion using llama.cpp asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the model.
        - system (str, optional): System prompt to use for this completion.

        Yields:
        - str: The content of each chunk from the model response.
        """
        logger.debug(f"Generating completion for messages: {messages}")

        try:
            # Add system prompt if provided
            messages_with_system = messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *messages,
                ]

            # Create chat completion in a separate thread to avoid blocking
            chat_completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(
                    messages=messages_with_system,
                    stream=True,
                ),
            )

            # Process chunks
            for chunk in chat_completion:
                if chunk.get("choices") and chunk["choices"][0].get("delta"):
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
