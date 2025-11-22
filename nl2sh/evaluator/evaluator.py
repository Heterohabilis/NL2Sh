# we use gpt-5.1 as the judge: cheap while powerful
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nl2sh.agents.llm_service import LLMService
from nl2sh.prompts.eval_pmpt import eval_prompt
from typing import Dict, Any, List, Tuple
import json
from tqdm.auto import tqdm

class Evaluator:
    def __init__(self, model: str = 'gpt-5.1'):
        self.model = model
        self.template = eval_prompt
        self.instance = LLMService(model)

    def _eval_one(self, task: str, command: str) -> float:
        prompt = ((self.template
                  .replace("{{TASK_DESCRIPTION}}", task))
                  .replace("{{BASH_COMMAND}}", command))
        prompt_set = [
            {"role": "user", "content": prompt},
        ]
        res = self.instance.chat(prompt_set)
        if not res:
            raise ValueError("The LLM said nothing")
        return float(res.strip())

    def eval_batch(
            self,
            pairs: List[Tuple[str, str]],
            num_workers: int = 5,
            ofile: str | Path | None = None,
    ) -> List[Tuple[str, str, int]]:
        results: List[Tuple[str, str, int]] = []
        total_score = 0
        if not pairs:
            return results

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_pair = {
                ex.submit(self._eval_one, task, cmd): (task, cmd)
                for (task, cmd) in pairs
            }

            for fut in tqdm(
                    as_completed(future_to_pair),
                    total=len(future_to_pair),
                    desc=f"Judging commands (workers={num_workers})",
                    unit="case",
            ):
                task, cmd = future_to_pair[fut]
                try:
                    score = fut.result()
                except Exception as e:
                    print(f"[WARN] judging failed for task: {task!r}, cmd: {cmd!r}, err: {e}")
                    score = -1
                results.append((task, cmd, score))
                total_score+=score

        if ofile is not None:
            ofile = Path(ofile)
            with ofile.open("w", encoding="utf-8") as f:
                for task, cmd, score in results:
                    rec: Dict[str, Any] = {
                        "task": task,
                        "command": cmd,
                        "score": score,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # f.write(f"Total score: {total_score}, avg_score = {total_score / len(results)}\n")
            print(f"total score: {total_score}, avg_score = {total_score / len(results)}")
            print(f"Saved {len(results)} judged records to {ofile}")

        return results

    def eval_from_file(
            self,
            infile: str | Path,
            outfile: str | Path,
            num_workers: int = 5,
    ) -> List[Tuple[str, str, int]]:
        infile = Path(infile)
        pairs: List[Tuple[str, str]] = []

        with infile.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                task = obj.get("task", "")
                cmd = obj.get("command", "")
                if task and cmd:
                    pairs.append((task, cmd))

        return self.eval_batch(pairs, num_workers=num_workers, ofile=outfile)


if __name__ == "__main__":
    q = "print the number of processors"
    a = "task: cat /proc/cpuinfo | fgrep \"CPU(s)\" | head -n 1 | awk '{print $4}'| tac"
    ev = Evaluator()
    ev.eval_from_file(
        "gened_files/FT_ed_o.txt",
        "eval_results/FT_o_res.txt",
    )
