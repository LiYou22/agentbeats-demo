from .utils import *
from .eval_tasks import *
import ast
import argparse
import os
import json
from .personas import *
import logging
import re
from pathlib import Path

# -------------------------
# Agent-safe helper
# -------------------------
def agent_safe_fail(msg):
    """
    Never call exit() inside an agent.
    Raise instead so the agent server won't die.
    """
    raise FileNotFoundError(msg)

# -------------------------
# Path setup
# -------------------------
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent.parent

RUBRICS_DIR = PROJECT_ROOT / "rubrics"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
SCORES_DIR = PROJECT_ROOT / "scores"
full_rubrics_path = str(RUBRICS_DIR / "general")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------
# Model configs
# -------------------------
SETTINGS_MODEL = "gpt-4o-mini"
QUESTION_MODEL = "gpt-4o-mini"
EXAMPLE_MODEL = "gpt-4o-mini"
EVAL_1 = "gpt-4o-mini"
EVAL_2 = "gpt-4o"

# -------------------------
# Utilities
# -------------------------
def extract_list(original_string):
    list_string = original_string.replace("```python", "").replace("```", "")
    list_string = list_string.strip()
    return ast.literal_eval(list_string)

# -------------------------
# Core logic
# -------------------------
def select_settings(persona, settings_options):
    settings_prompt = f"""
    Given the following persona description, select the most relevant settings
    from the given settings options. Output ONLY a python list.

    Persona: {persona}
    Settings: {settings_options}
    Selected Settings:
    """
    selected_settings = run_model(
        input_prompt=settings_prompt,
        model_card=SETTINGS_MODEL
    )
    return extract_list(selected_settings)

def gen_questions(persona, settings, num_questions=1):
    questions = {task: [] for task in tasks}

    for task in tasks:
        description = question_requirements[task]
        question_prompt = f"""
        Generate exactly {num_questions} challenging multi-step questions.

        Persona: {persona}
        Settings: {settings}
        Evaluation Task: {task}
        Description: {description}

        Output ONLY a python list.
        """
        for _ in range(5):
            try:
                task_questions = run_model(
                    input_prompt=question_prompt,
                    model_card=QUESTION_MODEL
                )
                task_questions = extract_list(task_questions)
                if len(task_questions) == num_questions:
                    break
            except Exception:
                continue

        questions[task].extend(task_questions)

    return questions

def process_examples(text):
    matches = re.findall(
        r'Score (\d+): *Response - *"?(.*?)"?(?=\n*Score \d+: *Response -|$)',
        text,
        re.S
    )
    processed = '\n\n'.join(
        f'Score {s}: "{r.strip()}"' for s, r in matches
    )
    return "\n\n".join(
        line for line in processed.split("\n") if line.startswith("Score")
    )

def gen_score_examples(persona, qa, rubric, model):
    examples_rubric = open(PROMPTS_DIR / 'score_examples' / 'parallel_examples.txt').read()
    rubrics = []
    for question, _ in qa:
        score_prompt = open(PROMPTS_DIR / 'score_examples' / 'prompt.txt').read()
        score_prompt = score_prompt.format(persona = persona, question = question, rubric = rubric)
        rubrics.append(score_prompt)

    prompt = examples_rubric.format(rubrics=rubrics)

    examples = run_model(input_prompt=prompt, temperature=0, top_p=0, model_card=model)
    examples = process_examples(examples)
    return examples

def parse_rubric(text):
    match = re.search(r"final score is\s*(\d+)", text)
    return int(match.group(1)) if match else 0

def parse_evaluation_text(text):
    match = re.search(r"^(.*?)Therefore, the final score is", text, re.S)
    return match.group(1).strip() if match else text.strip()

def format_rubrics(persona, rubric, qa):
    sys_prompt = open(PROMPTS_DIR / 'rubric_grading' / 'sys_prompt.txt').read()
    prompt_outline = open(PROMPTS_DIR / 'rubric_grading' / 'prompt.txt').read()
    rubrics = []

    examples = gen_score_examples(persona, qa, rubric, EXAMPLE_MODEL)
    for i in range(len(qa)):
        question, answer = qa[i]
        score_examples = examples[i]
        formatted_rubric = rubric.format(persona = persona, question = question, response = answer, score_example = score_examples)
        rubrics.append(formatted_rubric)

    
    scoring_prompt = prompt_outline.format(rubrics = rubrics)

    return sys_prompt, scoring_prompt

def parse_evaluations(text):
    pattern = r'\(\d+\) Evaluation:(.*?)(?=\(\d+\) Evaluation:|$)'
    evaluations = re.findall(pattern, text, re.DOTALL)
    evaluations = [eval.strip() for eval in evaluations]
    return evaluations

def calculate_modified_average(scores):
    zeros = scores.count(0)
    denom = len(scores) - zeros
    return sum(scores) / denom if denom > 0 else sum(scores)

def score_rubrics(sys_prompt, scoring_prompt, num_evals=1, return_explanations=True):
    scores = []
    explanations = []

    for _ in range(num_evals):
        evaluator1 = run_model(input_prompt=scoring_prompt, temperature=0, top_p=0, model_card=EVAL_1, system = sys_prompt)
        evaluator2 = run_model(input_prompt=scoring_prompt, temperature=0, top_p=0, model_card=EVAL_2, system = sys_prompt)

        evaluator1 = parse_evaluations(evaluator1)
        evaluator2 = parse_evaluations(evaluator2)

        scores1 = [parse_rubric(rubric) for rubric in evaluator1]
        scores2 = [parse_rubric(rubric) for rubric in evaluator2]

        score1 = calculate_modified_average(scores1)
        score2 = calculate_modified_average(scores2)

        scores.append(score1)
        scores.append(score2)
        
        if return_explanations:
            # Parse evaluation explanations
            explanations1 = [parse_evaluation_text(eval_text) for eval_text in evaluator1]
            explanations2 = [parse_evaluation_text(eval_text) for eval_text in evaluator2]
            
            explanations.append({
                "evaluator1": {
                    "scores": scores1,
                    "explanations": explanations1
                },
                "evaluator2": {
                    "scores": scores2,
                    "explanations": explanations2
                }
            })
    
    if return_explanations:
        return {
            "average_score": sum(scores) / len(scores),
            "detailed_explanations": explanations
        }
    else:
        return sum(scores) / len(scores)

def gen_answers(persona, questions, model):
    task_to_qa = {}
    for task, qs in questions.items():
        task_to_qa[task] = []
        for q in qs:
            a = run_model(input_prompt=q, persona=persona, model_card=model)
            task_to_qa[task].append((q, a))
    return task_to_qa

def score_answers(
    persona,
    task_to_qa,
    rubrics_path=full_rubrics_path,
    return_explanations=True
):
    result = {task: {"scores": [], "reasons": []} for task in task_to_qa}

    for task, qa_list in task_to_qa.items():
        if not qa_list:
            continue

        rubric_file = f"{rubrics_path}/{task}.txt"
        if not os.path.exists(rubric_file):
            logger.warning(f"Missing rubric: {rubric_file}")
            continue

        with open(rubric_file) as f:
            rubric = f.read()

        for i in range(0, len(qa_list), 5):
            batch = qa_list[i:i + 5]
            sys_prompt, scoring_prompt = format_rubrics(persona, rubric, batch)

            scores = []
            explanations = []

            for model in [EVAL_1, EVAL_2]:
                eval_text = run_model(
                    input_prompt=scoring_prompt,
                    system=sys_prompt,
                    temperature=0,
                    top_p=0,
                    model_card=model
                )
                evals = re.findall(r'\(\d+\) Evaluation:(.*?)(?=\(\d+\)|$)', eval_text, re.S)
                scores.extend(parse_rubric(e) for e in evals)
                explanations.extend(parse_evaluation_text(e) for e in evals)

            result[task]["scores"].append(calculate_modified_average(scores))
            if return_explanations:
                result[task]["reasons"].extend(explanations)

    return result

def save_responses(persona, task_to_qa, model_name):
    dir = RESULTS_DIR / model_name
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(dir / f'{persona}_qa.json', 'w') as file:
        json.dump(task_to_qa, file, indent=4)

def save_scores(save_name, scores):
    dir = SCORES_DIR / save_name
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Issac: I changed to save the json directly.
    with open(dir / 'scores.json', 'w') as file:
        json.dump(scores, file, indent=4)

def load_questions(persona, saved_questions):
    base_questions_dir = PROJECT_ROOT / "questions"
    dir = base_questions_dir / saved_questions
    if not os.path.exists(dir):
        print(f"No questions directory {dir}")
        exit(0)
    
    file_path = f'{dir}/{persona}.json'
    if not os.path.exists(file_path):
        print(f"No JSON file {file_path}")
        exit(0)

    with open(file_path, 'r') as file:
        questions = json.load(file)

    return questions

def load_responses(persona, saved_responses): 
    dir = saved_responses
    if not os.path.exists(dir):
        print(f"No responses directory {saved_responses}")
        exit(0)
    
    file_path = f'{dir}/{persona}_qa.json'
    if not os.path.exists(file_path):
        print(f"No JSON file {file_path}")
        exit(0)

    with open(file_path, 'r') as file:
        task_to_qa = json.load(file)

    return task_to_qa

# -------------------------
# Agent-safe entry
# -------------------------
def main(
    persona,
    model,
    model_name=None,
    saved_questions=None,
    saved_responses=None,
    return_explanations=True
):
    try:
        if saved_responses:
            if not os.path.exists(saved_responses):
                agent_safe_fail(f"No responses directory {saved_responses}")
            task_to_qa = load_responses(persona, saved_responses)

        else:
            if saved_questions:
                task_to_qa = load_questions(persona, saved_questions)
            else:
                settings = select_settings(persona, settings_list)
                questions = gen_questions(persona, settings)
                task_to_qa = gen_answers(persona, questions, model)

        result = score_answers(persona, task_to_qa, return_explanations=return_explanations)

        overall_scores = [
            sum(v["scores"]) / len(v["scores"])
            for v in result.values()
            if v["scores"]
        ]

        if overall_scores:
            result["PersonaScore"] = {
                "scores": [sum(overall_scores) / len(overall_scores)],
                "reasons": []
            }

        return result

    except Exception as e:
        logger.exception("Agent-safe failure")
        return {
            "PersonaScore": {
                "scores": [],
                "reasons": [str(e)]
            }
        }

# -------------------------
# CLI (unchanged)
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_list", type=str, default="[]")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--benchmark", type=str, default=None)

    args = parser.parse_args()
    personas = eval(args.persona_list)

    results = {}
    for p in personas:
        results[p] = main(p, args.model)

    logger.info(results)
