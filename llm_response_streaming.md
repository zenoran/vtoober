# How the App Splits and Streams LLM Responses

## Overview
The app uses the OpenAI API to generate language model responses asynchronously. The response is split into chunks, which are then streamed back to the client. This allows for real-time processing and display of the generated text.

## Key Components

### 1. Stream Initialization
The `chat_completion` method in `openai_compatible_llm.py` initializes a stream with the OpenAI API.

```python
stream: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
    messages=messages_with_system,
    model=self.model,
    stream=True,
    temperature=self.temperature,
    tools=available_tools,
)
```

### 2. Streaming and Yielding Content
The app processes each chunk in the stream, checking for tool calls and regular content.

```python
async for chunk in stream:
    if self.support_tools:
        has_tool_calls = (
            hasattr(chunk.choices[0].delta, "tool_calls")
            and chunk.choices[0].delta.tool_calls
        )

        # Process tool calls if any
        if has_tool_calls:
            logger.debug(f"Tool calls detected in chunk: {chunk.choices[0].delta.tool_calls}")
            in_tool_call = True
            for tool_call in chunk.choices[0].delta.tool_calls:
                index = tool_call.index if hasattr(tool_call, "index") else 0

                # Initialize or update tool call information
                if index not in accumulated_tool_calls:
                    accumulated_tool_calls[index] = {
                        "index": index,
                        "id": getattr(tool_call, "id", None),
                        "type": getattr(tool_call, "type", None),
                        "function": {"name": "", "arguments": ""},
                    }

                # Update tool call details
                if hasattr(tool_call, "id") and tool_call.id:
                    accumulated_tool_calls[index]["id"] = tool_call.id
                if hasattr(tool_call, "type") and tool_call.type:
                    accumulated_tool_calls[index]["type"] = tool_call.type

                if hasattr(tool_call, "function"):
                    if hasattr(tool_call.function, "name") and tool_call.function.name:
                        accumulated_tool_calls[index]["function"]["name"] = tool_call.function.name
                    if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                        accumulated_tool_calls[index]["function"]["arguments"] += tool_call.function.arguments

            continue

        # Yield accumulated tool calls if no more tool calls in the current chunk
        elif in_tool_call and not has_tool_calls:
            in_tool_call = False
            complete_tool_calls = [
                ToolCallObject.from_dict(tool_data)
                for tool_data in accumulated_tool_calls.values()
            ]
            yield complete_tool_calls
            accumulated_tool_calls = {}

    # Process regular content chunks
    if len(chunk.choices) == 0:
        logger.info("Empty chunk received")
        continue
    elif chunk.choices[0].delta.content is None:
        chunk.choices[0].delta.content = ""

    content = chunk.choices[0].delta.content
    yield content

# Yield any remaining tool calls at the end of the stream
if in_tool_call and accumulated_tool_calls:
    complete_tool_calls = [
        ToolCallObject.from_dict(tool_data)
        for tool_data in accumulated_tool_calls.values()
    ]
```

### 3. Explanation
- **Stream Initialization**: The `chat_completion` method initializes a stream with the OpenAI API.
- **Streaming and Yielding Content**:
  - The app processes each chunk in the stream.
  - If tool calls are detected, they are accumulated and yielded separately.
  - Regular content chunks are yielded immediately as they are received.

## Conclusion
The app splits the LLM response into chunks and streams them using an asynchronous generator. Each chunk is processed to check for tool calls, and if any are present, they are accumulated and yielded separately. Regular content chunks are also yielded as they are received.