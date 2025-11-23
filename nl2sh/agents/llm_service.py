"""
    LLM service wrapper for OpenAI API
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any
import json

"""
As a convention, we save the key in the .env file as OPENAI_KEY in the project root directory.
We load it here and save as a global constant KEY.
"""
load_dotenv()
KEY = os.getenv("OPENAI_KEY")


class LLMService:
    """  
    LLM service wrapper for OpenAI API
    Uses the OpenAI Python SDK to interact with the API.
    Attributes:
        model (str): The model to use for the LLM service.
        client (OpenAI): The OpenAI client instance.
    Methods:
        chat(messages: List[Dict[str, Any]]) -> str: Sends a chat request to the LLM service and returns the response as a string.
        chat_json(messages: List[Dict[str, Any]]) -> Any: Sends a chat request to the LLM service and returns the response parsed as JSON.
    """

    def __init__(self, model = "gpt-4-mini") -> None:
        self.client = OpenAI(api_key=KEY)
        self.model = model

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """
        messages: e.g.
        [
            {"role": "system", "content": "You are ..."},
            {"role": "user", "content": "Hello"},
        ]
        """
        resp = self.client.responses.create(
            model=self.model,
            input=messages,
        )
        return resp.output_text

    def chat_json(self, messages: List[Dict[str, Any]]) -> Any:
        text = self.chat(messages)
        return json.loads(text)


if __name__ == "__main__":
    # ft:gpt-4o-mini-2024-07-18:personal:dl-prj-2-750-filtered:CeGCZAoF
    agent = LLMService(model = "ft:gpt-4o-mini-2024-07-18:personal:dl-prj-2-750-filtered:CeGCZAoF")
    o = agent.chat(messages = [
    {"role": "system", "content": "You are an expert Linux Bash assistant. Translate the user's natural language request into a valid Bash command. Output only the command code without markdown or explanation."},
    {"role": "user", "content": "Count all the lines of all files with names ending with 'php' in current directory and subdirectories recursively"},
],)
    print(o)
