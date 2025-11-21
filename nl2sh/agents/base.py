import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any
import json

load_dotenv()
KEY = os.getenv("OPENAI_KEY")

class LLM:
    def __init__(self, model = "gpt-5-nano") -> None:
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
    # ft:gpt-4o-mini-2024-07-18:personal:dl-prj-2:CeF0gA8R
    agent = LLM(model = "gpt-4o")
    o = agent.chat(messages = [
    {"role": "system", "content": "You are an expert Linux Bash assistant. Translate the user's natural language request into a valid Bash command. Output only the command code without markdown or explanation."},
    {"role": "user", "content": "Count all the lines of all files with names ending with 'php' in current directory and subdirectories recursively"},
],)
    print(o)
