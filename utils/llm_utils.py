"""
LLM utilities for FINCON system.
Functions for interacting with large language models.
"""

import os
import logging
import openai
import time
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger("llm_utils")

class LLMClient:
    """
    Client for interacting with large language models.
    
    Provides consistent interfaces for different LLM backends
    and includes retry logic for API calls.
    """
    
    def __init__(self, model_name=None, api_key=None, temperature=0.3, max_tokens=1024):
        """
        Initialize LLM client.
        
        Args:
            model_name (str, optional): Name of the LLM model to use
            api_key (str, optional): API key for accessing the model
            temperature (float): Temperature parameter for sampling
            max_tokens (int): Maximum tokens to generate
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4-turbo")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt, system_message=None, temperature=None):
        """
        Generate text using the LLM.
        
        Args:
            prompt (str): Input prompt for generation
            system_message (str, optional): System message for chat models
            temperature (float, optional): Override default temperature
            
        Returns:
            str: Generated text
        """
        try:
            # Use either provided temperature or default
            temp = temperature if temperature is not None else self.temperature
            
            # Prepare messages for chat models
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens
            )
            end_time = time.time()
            
            # Log timing
            logger.debug(f"LLM request took {end_time - start_time:.2f} seconds")
            
            # Extract generated text
            generated_text = response.choices[0].message.content.strip()
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text from LLM: {str(e)}")
            raise
            
    def generate_with_format(self, prompt, output_format, system_message=None, temperature=None):
        """
        Generate text with a specific output format.
        
        Args:
            prompt (str): Input prompt for generation
            output_format (str): Description of required output format
            system_message (str, optional): System message for chat models
            temperature (float, optional): Override default temperature
            
        Returns:
            str: Generated text in specified format
        """
        formatted_prompt = f"""{prompt}

Please format your response as follows:
{output_format}"""

        return self.generate(formatted_prompt, system_message, temperature)
        
    def generate_json(self, prompt, json_schema, system_message=None, temperature=None):
        """
        Generate JSON output using the LLM.
        
        Args:
            prompt (str): Input prompt for generation
            json_schema (dict): Schema of the required JSON output
            system_message (str, optional): System message for chat models
            temperature (float, optional): Override default temperature
            
        Returns:
            dict: Generated JSON object
        """
        # Add JSON schema to prompt
        formatted_prompt = f"""{prompt}

Please provide your response as a valid JSON object with the following schema:
```json
{json.dumps(json_schema, indent=2)}
```

Your response should only contain the JSON object, without any additional text."""

        # Generate response
        response = self.generate(formatted_prompt, system_message, temperature)
        
        # Extract and parse JSON
        try:
            # Try to find JSON in the response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].strip()
            else:
                json_text = response.strip()
                
            # Parse JSON
            result = json.loads(json_text)
            return result
            
        except Exception as e:
            logger.error(f"Error parsing JSON from LLM response: {str(e)}")
            logger.error(f"Raw response: {response}")
            
            # Return empty dict on error
            return {}
            
    def get_embedding(self, text, model="text-embedding-ada-002"):
        """
        Get embedding for text.
        
        Args:
            text (str): Text to embed
            model (str): Embedding model to use
            
        Returns:
            list: Embedding vector
        """
        try:
            response = openai.Embedding.create(
                input=[text],
                model=model
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return empty embedding on error
            return [0.0] * 1536  # Default size for OpenAI embeddings