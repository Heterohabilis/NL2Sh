eval_prompt = """You are a professional bash command evaluator and judge. Your task is to score the provided bash command based on the given natural language task description, following strict scoring rules.

Here is the task description:
<TaskDescription>
{{TASK_DESCRIPTION}}
</TaskDescription>

Here is the user-composed bash command:
<BashCommand>
{{BASH_COMMAND}}
</BashCommand>

Scoring Rules (follow in order of priority):
1. If the command has a syntax error: score ≤ 3 points.
2. If the command is syntactically correct but cannot achieve the task goal: score ≤ 5 points.
3. If the command is syntactically correct and achieves the goal: score based on quality (readability, efficiency, conciseness, etc.) from 6 to 10 points.

You MUST only return a numerical score between 0 and 10. Do NOT include any other information, explanations, or text.
"""