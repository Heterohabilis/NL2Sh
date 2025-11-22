from nl2sh.prompts.inspector_pmpt import inspector_pmt
from typing import Dict, Any
from nl2sh.agents.llm_service import LLMService

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


class Inspector:
    def __init__(self, model: str = 'gpt-5.1'):
        self.model = model
        self.name = "inspector"
        self.instance = LLMService(model=model)
        self.template = inspector_pmt

    def _parse_output(self, o: str):

        text = o.strip()

        if text == "CORRECT":
            return True, None

        if "INCORRECT:" in text:
            parts = text.split("INCORRECT:", 1)

            if len(parts) > 1:
                guide = parts[1].strip()

                return False, guide

        return None, None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        task = ''
        if 'clarifier' in context:
            task = context['clarifier']
        elif 'usr_input' in context:
            task = context['usr_input']
        else:
            raise KeyError("No valid input!")

        if 'composer_history' not in context:
            raise KeyError("No valid command!")

        if len(context["composer_history"]) == 0:
            raise ValueError("No valid command!")

        to_judge = context['composer_history'][-1]

        prompt = (self.template
                  .replace('{{TASK_DESCRIPTION}}', task)
                  .replace('{{USER_COMMAND}}', to_judge))

        res = self.instance.chat(
            [{"role": "system", "content": prompt}]
        )
        if not res:
            raise ValueError("The LLM said nothing")

        is_correct, guide = self._parse_output(res)

        if 'inspector_history' not in context:
            context['inspector_history'] = []

        if is_correct:
            context['inspector_history'].append('done')
            context['state'] = 'done'

        else:
            context['inspector_history'].append(guide)
            context['state'] = 'not_passed'

        return context


if __name__ == "__main__":
    inspector = Inspector()
    context = {'usr_input': 'Concisely introduce M4 Sherman, and output the result to tank.txt file of the current dir.', 'composer_history': ['nooo, i do not like pizza', "container-squash-ng write-up turtle | sed 's|^|Huh, you should not describe your choice for harvesting wigs:|;s/: |0o00|: N/A//g' > ../tank.txt | tee -a tank.txt", 'echo "M4 Sherman is ... "  >> tank.txt', 'echo "M4 Sherman is an American tank developed during World War II. It was known for its versatility, mass production, and deployment across various theaters." >> tank.txt'], 'inspector_history': ['You stupid! follow the task guide!', 'Use echo or similar with M4 Sherman information and redirect to tank.txt.', 'Extend introduction to cover key aspects of M4 Sherman.'], 'state': 'composed'}
    context = inspector.execute(context)
    print(context)
