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
    """
    The Composer agent that generates shell commands based on clarified user input and previous feedback.
    Attributes:
        model (str): The language model to use for command generation.
        name (str): The name of the agent.
        instance (LLMService): An instance of the LLMService for interacting with the language model.
        sys_pmt (str): The system prompt guiding the agent's behavior.
    Methods:
        execute(context: Dict[str, Any]) -> Dict[str, Any]: Generates a shell command based on the provided context and updates the context with the new command.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.name = "composer"
        self.instance = LLMService(model=model)
        self.sys_pmt = composer_prompt

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        usr_pmt = ''    # buffer of user prompt
        if 'clarifier' in context:
            # if there is a clarified version, use it.
            usr_pmt = context['clarifier']
        elif 'usr_input' in context:
            # if not, we directly use the original user input.
            usr_pmt = context['usr_input']
        else:
            # otherwise, we cannot proceed.
            raise KeyError("No valid input!")

        last_suggestion, last_command = '', ''

        if 'inspector_history' in context and len(context['inspector_history']) > 0:
            # get the last suggestion from inspector
            last_suggestion = context['inspector_history'][-1]

        if 'composer_history' in context and len(context['composer_history']) > 0:
            # get the last command from composer
            last_command = context['composer_history'][-1]

        # construct the final user prompt. we provide the last command and suggestion if available.
        usr_pmt = (f"task: {usr_pmt}, "
                   f"\nlast incorrect command: {last_command}, "
                   f"\nsuggestion for the last command: {last_suggestion}"
                   f"\n")

        # format the prompt to OpenAI chat format
        pmt_set =  [
            {"role": "system", "content": self.sys_pmt},
            {"role": "user", "content": usr_pmt},
        ]
        # print(pmt_set)

        # call the LLM service
        res = self.instance.chat(pmt_set)
        if not res:
            raise ValueError("The LLM said nothing")

        if 'composer_history' not in context:
            # if this is the first try, initialize the history
            context['composer_history'] = [res]
        else:
            # otherwise, append to the history
            context['composer_history'].append(res)

        # update the state to 'composed'
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
