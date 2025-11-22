from nl2sh.agents.llm_service import LLMService
from nl2sh.prompts.composer_pmpt import composer_prompt
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

class Composer:
    # can be ft:gpt-4o-mini-2024-07-18:personal:dl-prj-2-750-filtered:CeGCZAoF
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.name = "composer"
        self.instance = LLMService(model=model)
        self.sys_pmt = composer_prompt

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        usr_pmt = ''
        if 'clarifier' in context:
            usr_pmt = context['clarifier']
        elif 'usr_input' in context:
            usr_pmt = context['usr_input']
        else:
            raise KeyError("No valid input!")

        last_suggestion, last_command = '', ''

        if 'inspector_history' in context and len(context['inspector_history']) > 0:
            last_suggestion = context['inspector_history'][-1]

        if 'composer_history' in context and len(context['composer_history']) > 0:
            last_command = context['composer_history'][-1]

        usr_pmt = (f"task: {usr_pmt}, "
                   f"\nlast incorrect command: {last_command}, "
                   f"\nsuggestion for the last command: {last_suggestion}"
                   f"\n")

        pmt_set =  [
            {"role": "system", "content": self.sys_pmt},
            {"role": "user", "content": usr_pmt},
        ]
        # print(pmt_set)

        res = self.instance.chat(pmt_set)
        if not res:
            raise ValueError("The LLM said nothing")

        if 'composer_history' not in context:
            context['composer_history'] = [res]
        else:
            context['composer_history'].append(res)

        context['state'] = 'composed'
        return context


if __name__ == "__main__":
    model = "gpt-4o-mini"
    #model = "ft:gpt-4o-mini-2024-07-18:personal:dl-prj-2-1k-filtered:CeTXCBeh"
    context = {
        "usr_input": "Concisely introduce M4 Sherman, and output the result to tank.txt file of the current dir.",
        "composer_history": ['nooo, i do not like pizza'],
        "inspector_history": ['You stupid! follow the task guide!'],
    }
    c = Composer(model)
    context = c.execute(context)
    print(context)
