import json
import os
from pathlib import Path
from typing import List

from openai.resources import FineTuning
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


class Inference:
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

    def run_single(self, task: str, max_recompose: int | None = None) -> str:
        print(f"Current Task: {task} \n {'='*64}")
        context = {
            "usr_input": task,
            "clarifier": "",
            "composer_history": [],
            "inspector_history": [],
            "state": INIT,
        }

        recompose_cnt = 0

        while context["state"] != DONE:
            curr_state = context["state"]

            if curr_state == NOT_PASS and max_recompose is not None:
                if recompose_cnt >= max_recompose:
                    print("\n[Warning] Maximum recomposition attempts reached")
                    break
                recompose_cnt += 1

            next_agent = self.sched[curr_state]
            print(
                f"\n[State]   {curr_state}\n"
                f"[Next]    {next_agent.name}\n"
                f"{'-' * 64}"
            )
            try:
                context = next_agent.execute(context)
            except Exception as e:
                raise RuntimeError(f"something wrong with the inference: {e}")

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
            return final_cmd
        else:
            return ""


    def gen_eval_commands(self, tasks: List[str],
                          max_recompose: int | None = None,
                          ofile: str|None = None) -> List[tuple[str, str]]:
        results: List[tuple[str, str]] = []
        for task in tqdm(tasks, desc="Evaluating tasks", unit="task"):
            cmd = self.run_single(task, max_recompose)
            results.append((task, cmd))
        if not ofile:
            print(results)
        else:
            with open(ofile, "w", encoding="utf-8") as f:
                for task, cmd in results:
                    record = {"task": task, "command": cmd}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved {len(results)} records to {ofile}")
        return results

    def load_validation_nl(self,
            path: str | Path = "nl2sh/data/nl2bash_validation_50.jsonl",
    ) -> List[str]:

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


class TryMe:
    pass

if __name__ == "__main__":
    e = Inference(inspect_abltn=True)
    test_set = e.load_validation_nl()
    e.gen_eval_commands(test_set, max_recompose=2, ofile = './gened_files/base_o.txt')