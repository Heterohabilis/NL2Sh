import json
from pathlib import Path
from typing import List, Any

from tqdm import tqdm

from nl2sh.agents.clarifier import Clarifier
from nl2sh.agents.composer import Composer
from nl2sh.agents.inspector import Inspector


# States
INIT = 'init'
CLARIFIED = 'clarified'
COMPOSED = 'composed'
NOT_PASS = 'not_passed'
DONE = 'done'

FT_MD = 'ft:gpt-4o-mini-2024-07-18:personal:dl-prj-2-1k-filtered:CeTXCBeh'
MD = 'gpt-4o-mini'


def load_evaluation_nl(path: str | Path = "nl2sh/data/nl2bash_eval_50.jsonl",
                       ) -> List[str]:
    """
    Load evaluation NL tasks from a JSONL file. Each line in the file should be a JSON object
    containing a "messages" field, which is a list of message objects. The function extracts
    the content of the first message with the role "user" from each JSON object.
    Args:
        path (str | Path): Path to the JSONL file containing evaluation tasks.
    Returns:
        List[str]: A list of NL-to-shell tasks extracted from the file.
    """

    path = Path(path)
    nl_list: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {line_no}: JSON decode error: {e}")
                continue

            messages = obj.get("messages", [])
            user_msgs = [
                m.get("content", "")
                for m in messages
                if m.get("role") == "user"
            ]

            if not user_msgs:
                print(f"[WARN] line {line_no}: no user message found")
                continue

            nl_list.append(user_msgs[0])

    return nl_list


class Inference:
    """
    Inference pipeline for NL2SH task, integrating Clarifier, Composer, and Inspector agents.
    Attributes:
        composer (Composer): The Composer agent for generating shell commands.
        clarifier (Clarifier): The Clarifier agent for refining user input.
        inspector (Inspector): The Inspector agent for validating generated commands.
        sched (dict): A scheduling dictionary mapping states to agents to implement the finite state machine.
    Methods:
        run_single(task: str, max_recompose: int | None = None) -> tuple[str | Any, int] | str:
            Runs the inference pipeline for a single NL task.
        gen_eval_commands(tasks: List[str], max_recompose: int | None = None, ofile: str|None = None) -> List[tuple[str, str, int]]:
            Generates shell commands for a list of NL tasks and optionally saves the results to a file. 
    Finite State Machine States:
        INIT: Initial state before any processing.
        CLARIFIED: State after the Clarifier has refined the user input.
        COMPOSED: State after the Composer has generated a shell command.
        NOT_PASS: State when the Inspector does not approve the generated command.
        DONE: Final state indicating successful completion of the pipeline.
    State Transitions:
        init -> clarifier -> clarified
        clarified -> composer -> composed
        [composed -> inspector -> done / not_pass
        not_pass -> composer -> composed] repeat until done
    """

    def __init__(self, use_finetune: bool=False, inspect_abltn: bool=False):
        self.composer = Composer(model = FT_MD) if use_finetune else Composer()
        self.clarifier = Clarifier()
        self.inspector = Inspector(MD) if inspect_abltn else Inspector()
        self.sched = {
            INIT: self.clarifier,
            CLARIFIED: self.composer,
            COMPOSED: self.inspector,
            NOT_PASS: self.composer,
        }
        print(f"Current model settings: \n {'='*64} \n"
              f"Composer = {self.composer.model} \n"
              f"Clarifier = {self.clarifier.model} \n"
              f"Inspector = {self.inspector.model}")

    def run_single(self, task: str, max_recompose: int | None = None) -> tuple[str | Any, int] | str:
        """
        Run the inference pipeline for a single NL task.
        Args:
            task (str): The NL task to be processed.
            max_recompose (int | None): Maximum number of recomposition attempts if the inspector does not pass.
        Returns:
            tuple[str | Any, int] | str: The final generated shell command and the number of recomposition attempts or an empty string if no command was generated.
        """

        print(f"Current Task: {task} \n {'='*64}")

        # init the context
        context = {
            "usr_input": task,
            "clarifier": "",
            "composer_history": [],
            "inspector_history": [],
            "state": INIT,
        }
        
        # recompose counter
        recompose_cnt = 0

        # main loop
        while context["state"] != DONE:

            # get current state
            curr_state = context["state"]

            # check max recompose attempts
            if curr_state == NOT_PASS and max_recompose is not None:
                if recompose_cnt >= max_recompose:
                    print("\n[Warning] Maximum recomposition attempts reached")
                    break
                recompose_cnt += 1

            # retrieve next agent
            next_agent = self.sched[curr_state]

            # print state info
            print(
                f"\n[State]   {curr_state}\n"
                f"[Next]    {next_agent.name}\n"
                f"{'-' * 64}"
            )
            try:
                # based on the design, each agent has an execute function
                context = next_agent.execute(context)
            except Exception as e:
                raise RuntimeError(f"something wrong with the inference: {e}")

        # final report
        if context["composer_history"]:
            final_cmd = context["composer_history"][-1]
        else:
            final_cmd = "<no command generated>"

        if context["state"] == DONE:
            state_note = "SUCCESS"
        elif context["state"] == NOT_PASS:
            state_note = "INCOMPLETE (inspector did not pass)"
        else:
            state_note = "INTERRUPTED"

        # beautiful print
        print(
            "\n\n" +
            "=" * 64 + "\n"
            " NL2SH Evaluation Report\n"
            + "=" * 64 + "\n"
            f"User Input        : {context['usr_input']}\n"
            f"Final State       : {context['state']}  [{state_note}]\n"
            f"Recompose Attempts: {recompose_cnt}"
            + (f" / {max_recompose}" if max_recompose is not None else "")
            + "\n"
            f"Final Command     : {final_cmd}\n"
            + "=" * 64
        )

        if context["composer_history"]:
            return final_cmd, recompose_cnt
        else:
            return ""


    def gen_eval_commands(self, tasks: List[str],
                          max_recompose: int | None = None,
                          ofile: str|None = None) -> List[tuple[str, str, int]]:
        """
        Generate shell commands in batch for a list of NL tasks and optionally save the results to a file.
        Args:
            tasks (List[str]): A list of NL tasks to be processed.
            max_recompose (int | None): Maximum number of recomposition attempts if the inspector does not pass.
            ofile (str | None): Optional output file path to save the results in JSONL format.
        Returns:
            List[tuple[str, str, int]]: A list of tuples containing the NL task, generated shell command, and number of recomposition attempts.
        """
        results: List[tuple[str, str, int]] = []    # structure: (task, command, retry_times)

        # to avoid race condition, we run sequentially here
        for task in tqdm(tasks, desc="Evaluating tasks", unit="task"):
            cmd, retry_times = self.run_single(task, max_recompose)
            results.append((task, cmd, retry_times))
        
        # if no output file, print to console
        if not ofile:
            print(results)
        
        # else, save to the specified file
        else:
            with open(ofile, "w", encoding="utf-8") as f:
                for task, cmd, retry_times in results:
                    record = {"task": task, "command": cmd, "retry_times": retry_times}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved {len(results)} records to {ofile}")
        return results


"""
    class TryMe:
        pass
"""

if __name__ == "__main__":
    e = Inference(inspect_abltn=True)
    test_set = load_evaluation_nl()
    e.gen_eval_commands(test_set, max_recompose=2, ofile = './gened_files/base_o.txt')