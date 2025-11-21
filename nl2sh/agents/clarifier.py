from nl2sh.agents.llm_service import LLMService
from nl2sh.prompts.clarifier_pmpt import clarifier_prompt
from typing import Dict, Any

"""
context = {
    "usr_input": "xxx", 
    "clarifier": "yyy",
    "composer_history": [
        'h1', 'h2', 'h3'
    ],
    "inspector_history": [
        'h1', 'h2', 'h3'
    ],
    "state": "sss"
}
"""


class Clarifier:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.name = "clarifier"
        self.instance = LLMService(model=model)
        self.template = clarifier_prompt

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if "usr_input" not in context:
            raise KeyError("Missing usr_input in context")

        usr_input = context["usr_input"]
        prompt = self.template.replace("{{USER_NATURAL_LANGUAGE_REQUEST}}", usr_input)

        res = self.instance.chat(
            [{"role": "user", "content": prompt}]
        )
        if not res:
            raise ValueError("The LLM said nothing")

        context["clarifier"] = res.strip()
        context["state"] = "clarified"
        return context


if __name__ == "__main__":
    # test with python -m nl2sh.agents.clarifier
    c = Clarifier(model = "gpt-4o-mini")
    context = {
        'usr_input': "Count all the line's md5 of all files with names ending with 'java' in current directory and subdirectories recursively"
    }
    context = c.execute(context)
    print(context)