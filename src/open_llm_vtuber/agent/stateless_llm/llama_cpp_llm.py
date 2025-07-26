"""Description: This file contains the implementation of the LLM class using llama.cpp.
This class provides a stateless interface to llama.cpp for language generation.
"""

import asyncio
from typing import AsyncIterator, List, Dict, Any, Optional, cast
from datetime import datetime
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from llama_cpp.llama_types import ChatCompletionChunk
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface


def strftime_now_function(format_string: str) -> str:
    """Jinja2 function to get current time formatted."""
    return datetime.now().strftime(format_string)


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
        
        self.model_path = model_path
        
        # Build parameters exactly like the working llama_cpp_client.py
        model_load_params = {
            "model_path": model_path,
            "n_gpu_layers": kwargs.get("n_gpu_layers", -1),
            "n_ctx": kwargs.get("n_ctx", 4096),
            "verbose": False,
        }
        
        # Add optional parameters if present
        if "n_batch" in kwargs:
            model_load_params["n_batch"] = kwargs["n_batch"]
        if "main_gpu" in kwargs:
            model_load_params["main_gpu"] = kwargs["main_gpu"]
        if "tensor_split" in kwargs and kwargs["tensor_split"] is not None:
            model_load_params["tensor_split"] = kwargs["tensor_split"]
        
        try:
            # Create a temporary Llama instance to get the chat format
            temp_llm = Llama(model_path=model_path, verbose=False)
            
            chat_template = temp_llm.metadata.get("tokenizer.chat_template")
            if not chat_template:
                raise ValueError("Could not find chat template in model metadata")

            eos_token = temp_llm.metadata.get("tokenizer.ggml.eos_token_id")
            bos_token = temp_llm.metadata.get("tokenizer.ggml.bos_token_id")

            # Create a chat formatter from the model config
            self.formatter = Jinja2ChatFormatter(
                template=chat_template, eos_token=str(eos_token), bos_token=str(bos_token)
            )

            # Add the custom function to the Jinja2 environment
            self.formatter._environment.globals["strftime_now"] = strftime_now_function
            logger.info("Injected 'strftime_now' into Jinja2 environment.")

            # We do not pass the chat_handler to Llama, as it interferes with streaming.
            # Instead, we will use it manually.

            # Log what we're actually passing to llama-cpp-python
            logger.info(f"llama-cpp-python parameters: {model_load_params}")

            self.llm = Llama(**model_load_params)
            
        except Exception as e:
            logger.critical(f"Failed to initialize Llama model: {e}")
            raise

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: Optional[str] = None
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
            # Preprocess messages to handle complex content structures
            processed_messages = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    # Extract text from the content list
                    text_content = " ".join(
                        item["text"] for item in msg["content"] if item.get("type") == "text"
                    )
                    processed_msg = msg.copy()
                    processed_msg["content"] = text_content
                    processed_messages.append(processed_msg)
                else:
                    processed_messages.append(msg)

            # Add system prompt if provided
            messages_with_system = processed_messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *processed_messages,
                ]

            # Manually format the prompt using the custom formatter
            prompt_response = self.formatter(messages=messages_with_system)
            prompt = prompt_response.prompt

            # Create completion in a separate thread to avoid blocking
            logger.debug("Calling create_completion with stream=True")
            completion_stream = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.create_completion(
                    prompt=prompt,
                    stop=[self.formatter.eos_token],
                    max_tokens=-1,
                    stream=True,
                ),
            )
            logger.debug(f"Received completion_stream of type: {type(completion_stream)}")

            # Process chunks
            for chunk_data in completion_stream:
                chunk = cast(dict, chunk_data)
                choices = chunk.get("choices")
                if choices and len(choices) > 0:
                    content = choices[0].get("text")
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise