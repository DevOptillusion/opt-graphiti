"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import re
import typing
from typing import TYPE_CHECKING, ClassVar

import httpx
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'grok-4-fast-reasoning'
DEFAULT_SMALL_MODEL = 'grok-4-fast-non-reasoning'
DEFAULT_MAX_TOKENS = 8192

# Maximum output tokens for different Grok models
GROK_MODEL_MAX_TOKENS = {
    'grok-4-fast-reasoning': 8192,
    'grok-4-fast-non-reasoning': 8192,
}

# Default max tokens for models not in the mapping
DEFAULT_GROK_MAX_TOKENS = 8192


class GrokClient(LLMClient):
    """
    GrokClient is a client class for interacting with xAI's Grok language models.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the Grok language model via the xAI API.

    Attributes:
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
    
    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: httpx.AsyncClient | None = None):
            Initializes the GrokClient with the provided configuration, cache setting, and optional HTTP client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2
    API_BASE_URL: ClassVar[str] = 'https://api.x.ai/v1'

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        """
        Initialize the GrokClient with the provided configuration, cache setting, and optional HTTP client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            max_tokens (int | None): The maximum number of tokens to generate. If None, uses model defaults.
            client (httpx.AsyncClient | None): An optional async HTTP client instance to use. If not provided, a new client is created.
        """
        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        self.model = config.model or DEFAULT_MODEL
        self.small_model = config.small_model or DEFAULT_SMALL_MODEL
        self.api_key = config.api_key
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS

        if client is None:
            self.client = httpx.AsyncClient(
                base_url=self.API_BASE_URL,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                timeout=60.0,
            )
        else:
            self.client = client

    def _get_model_for_size(self, model_size: ModelSize) -> str:
        """Get the appropriate model name based on the requested size."""
        if model_size == ModelSize.small:
            model = self.small_model or DEFAULT_SMALL_MODEL
            logger.info(f'[Grok] Using small_model: {model} (configured: {self.small_model}, default: {DEFAULT_SMALL_MODEL})')
            return model
        else:
            model = self.model or DEFAULT_MODEL
            logger.info(f'[Grok] Using model: {model} (configured: {self.model}, default: {DEFAULT_MODEL}, size: {model_size})')
            return model

    def _get_max_tokens_for_model(self, model: str) -> int:
        """Get the maximum output tokens for a specific Grok model."""
        return GROK_MODEL_MAX_TOKENS.get(model, DEFAULT_GROK_MAX_TOKENS)

    def _resolve_max_tokens(self, requested_max_tokens: int | None, model: str) -> int:
        """
        Resolve the maximum output tokens to use based on precedence rules.

        Precedence order (highest to lowest):
        1. Explicit max_tokens parameter passed to generate_response()
        2. Instance max_tokens set during client initialization
        3. Model-specific maximum tokens from GROK_MODEL_MAX_TOKENS mapping
        4. DEFAULT_GROK_MAX_TOKENS as final fallback

        Args:
            requested_max_tokens: The max_tokens parameter passed to generate_response()
            model: The model name to look up model-specific limits

        Returns:
            int: The resolved maximum tokens to use
        """
        # 1. Use explicit parameter if provided
        if requested_max_tokens is not None:
            return requested_max_tokens

        # 2. Use instance max_tokens if set during initialization
        if self.max_tokens is not None:
            return self.max_tokens

        # 3. Use model-specific maximum or return DEFAULT_GROK_MAX_TOKENS
        return self._get_max_tokens_for_model(model)

    def salvage_json(self, raw_output: str) -> dict[str, typing.Any] | None:
        """
        Attempt to salvage a JSON object if the raw output is truncated.

        This is accomplished by looking for the last closing bracket for an array or object.
        If found, it will try to load the JSON object from the raw output.
        If the JSON object is not valid, it will return None.

        Args:
            raw_output (str): The raw output from the LLM.

        Returns:
            dict[str, typing.Any]: The salvaged JSON object.
            None: If no salvage is possible.
        """
        if not raw_output:
            return None
        # Try to salvage a JSON array
        array_match = re.search(r'\]\s*$', raw_output)
        if array_match:
            try:
                return json.loads(raw_output[: array_match.end()])
            except Exception:
                pass
        # Try to salvage a JSON object
        obj_match = re.search(r'\}\s*$', raw_output)
        if obj_match:
            try:
                return json.loads(raw_output[: obj_match.end()])
            except Exception:
                pass
        return None

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Grok language model.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int | None): The maximum number of tokens to generate in the response. If None, uses precedence rules.
            model_size (ModelSize): The size of the model to use (small or medium).

        Returns:
            dict[str, typing.Any]: The response from the language model.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            Exception: If there is an error generating the response.
        """
        try:
            # Get the appropriate model for the requested size
            model = self._get_model_for_size(model_size)

            # Resolve max_tokens using precedence rules
            resolved_max_tokens = self._resolve_max_tokens(max_tokens, model)

            # Convert messages to Grok/OpenAI format
            grok_messages: list[dict[str, str]] = []
            system_prompt = ''

            # Handle system message if present
            if messages and messages[0].role == 'system':
                system_prompt = messages[0].content
                messages = messages[1:]

            # Add the rest of the messages
            for m in messages:
                m.content = self._clean_input(m.content)
                grok_messages.append({'role': m.role, 'content': m.content})

            if response_model is not None:
                messages[0].content += f"Please follow the format of the response_model: {response_model}"

            # Prepare request payload
            payload: dict[str, typing.Any] = {
                'model': model,
                'messages': grok_messages,
                'temperature': self.temperature,
                'max_tokens': resolved_max_tokens,
                'stream': False,
            }

            # Add system message if present
            if system_prompt:
                payload['messages'] = [{'role': 'system', 'content': system_prompt}] + grok_messages

            # Request JSON format if response_model is provided
            if response_model is not None:
                payload['response_format'] = {'type': 'json_object'}

            # Make API request
            response = await self.client.post('/chat/completions', json=payload)
            response.raise_for_status()

            response_data = response.json()

            # Extract content from response
            if 'choices' not in response_data or len(response_data['choices']) == 0:
                raise Exception('No choices in response from Grok API')

            choice = response_data['choices'][0]
            if 'message' not in choice or 'content' not in choice['message']:
                raise Exception('No content in response from Grok API')

            raw_output = choice['message']['content']

            # If this was a structured output request, parse the response into the Pydantic model
            if response_model is not None:
                try:
                    if not raw_output:
                        raise ValueError('No response text')

                    validated_model = response_model.model_validate(json.loads(raw_output))

                    # Return as a dictionary for API consistency
                    return validated_model.model_dump()
                except Exception as e:
                    if raw_output:
                        logger.error(
                            '🦀 LLM generation failed parsing as JSON, will try to salvage.'
                        )
                        logger.error(f"Error in generating Grok response: {e}")
                        # Try to salvage
                        salvaged = self.salvage_json(raw_output)
                        if salvaged is not None:
                            logger.warning('Salvaged partial JSON from truncated/malformed output.')
                            return salvaged
                    raise Exception(f'Failed to parse structured response: {e}') from e

            # Otherwise, return the response text as a dictionary
            return {'content': raw_output}

        except httpx.HTTPStatusError as e:
            # Check if it's a rate limit error
            if e.response.status_code == 429:
                raise RateLimitError from e
            error_message = str(e).lower()
            if 'rate limit' in error_message or 'quota' in error_message:
                raise RateLimitError from e
            logger.error(f'HTTP error in generating LLM response: {e}')
            raise Exception from e
        except Exception as e:
            # Check if it's a rate limit error based on error message
            error_message = str(e).lower()
            if (
                'rate limit' in error_message
                or 'quota' in error_message
                or '429' in str(e)
            ):
                raise RateLimitError from e

            logger.error(f'Error in generating LLM response: {e}')
            raise Exception from e
            
    # override the generate_response method in the base class to add extra format instruction for Grok Client
    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Grok language model with retry logic and error handling.
        This method overrides the parent class method to provide a direct implementation with advanced retry logic.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int | None): The maximum number of tokens to generate in the response.
            model_size (ModelSize): The size of the model to use (small or medium).
            group_id (str | None): Optional partition identifier for the graph.
            prompt_name (str | None): Optional name of the prompt for tracing.

        Returns:
            dict[str, typing.Any]: The response from the language model.
        """
        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nIMPORTANT: Your response MUST be ONLY a valid JSON object matching the schema below. Do not include any explanations, schemas, or additional text. Output nothing else:\n\n{serialized_model}'
            )
        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'grok',
                'model.size': model_size.value,
                'max_tokens': max_tokens or self.max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            retry_count = 0
            last_error = None
            last_output = None

            while retry_count < self.MAX_RETRIES:
                try:
                    response = await self._generate_response(
                        messages=messages,
                        response_model=response_model,
                        max_tokens=max_tokens,
                        model_size=model_size,
                    )
                    last_output = (
                        response.get('content')
                        if isinstance(response, dict) and 'content' in response
                        else None
                    )

                    # print("---****Grok Client messages****----")
                    # print(messages)
                    # print ("----****Grok Client response****--- ")
                    # print(response)
                    return response
                except RateLimitError as e:
                    # Rate limit errors should not trigger retries (fail fast)
                    span.set_status('error', str(e))
                    raise e
                except Exception as e:
                    last_error = e

                    retry_count += 1

                    # Construct a detailed error message for the LLM
                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {e.__class__.__name__}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid response, ensuring the output matches '
                        f'the expected format and constraints.'
                    )

                    error_message = Message(role='user', content=error_context)
                    messages.append(error_message)
                    logger.warning(
                        f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )

            # If we exit the loop without returning, all retries are exhausted
            logger.error('🦀 LLM generation failed and retries are exhausted.')
            logger.error(self._get_failed_generation_log(messages, last_output))
            logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {last_error}')
            span.set_status('error', str(last_error))
            span.record_exception(last_error) if last_error else None
            raise last_error or Exception('Max retries exceeded')

