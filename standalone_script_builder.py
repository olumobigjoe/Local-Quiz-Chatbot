"""
standalone_script_builder.py
-----------------------------
Builds a self-contained, dependency-free quiz-runner script that a user can
download and run completely offline, independent of this Streamlit app.

Two flavors are supported:
  - Python (runs in a terminal with `python quiz_runner.py`)
  - JavaScript (runs in Node.js with `node quiz_runner.js`)

The quiz data itself is embedded directly in the generated script as a
literal, so the downloaded file has zero external dependencies.
"""

import json
from typing import List, Dict, Any


def build_python_script(quiz_data: List[Dict[str, Any]]) -> str:
    """Return a fully commented, standalone Python script that runs the
    quiz interactively in a terminal."""
    quiz_json = json.dumps(quiz_data, indent=4)

    return f'''"""
quiz_runner.py
----------------
Auto-generated, self-contained quiz script.
No external dependencies required — pure Python standard library only.

Run it with:
    python quiz_runner.py
"""

import random

# The quiz questions, options, correct answers, and explanations are
# embedded below exactly as they were generated. Feel free to edit this
# list by hand if you want to tweak wording or add/remove questions.
QUIZ_DATA = {quiz_json}


def ask_question(index: int, item: dict) -> bool:
    """Print one question, collect the user's answer, and return whether
    it was correct."""
    print(f"\\nQ{{index + 1}}. {{item['question']}}")

    # Sort option labels so they display in a stable, readable order
    # (A, B, C... or I, II, III...) regardless of dict insertion order.
    for label in sorted(item["options"].keys()):
        print(f"  {{label}}) {{item['options'][label]}}")

    answer = input("Your answer: ").strip().upper()
    correct = item["correct"].strip().upper()

    if answer == correct:
        print("Correct!")
        return True
    else:
        print(f"Incorrect. The correct answer was {{item['correct']}}.")

    explanation = item.get("explanation")
    if explanation:
        print(f"Explanation: {{explanation}}")

    return False


def main():
    print("=" * 50)
    print("QUIZ TIME")
    print("=" * 50)
    print(f"This quiz has {{len(QUIZ_DATA)}} question(s).")

    # Ask if the user wants the question order shuffled, so repeat runs
    # of the same script don't always feel identical.
    shuffle_choice = input("Shuffle question order? (y/n): ").strip().lower()
    questions = list(QUIZ_DATA)
    if shuffle_choice == "y":
        random.shuffle(questions)

    score = 0
    for i, item in enumerate(questions):
        if ask_question(i, item):
            score += 1

    print("\\n" + "=" * 50)
    print(f"You scored {{score}} / {{len(questions)}}")
    print("=" * 50)


if __name__ == "__main__":
    main()
'''


def build_js_script(quiz_data: List[Dict[str, Any]]) -> str:
    """Return a fully commented, standalone Node.js script that runs the
    quiz interactively in a terminal."""
    quiz_json = json.dumps(quiz_data, indent=4)

    return f'''/**
 * quiz_runner.js
 * ----------------
 * Auto-generated, self-contained quiz script.
 * No external dependencies required — uses only Node.js's built-in
 * "readline" module.
 *
 * Run it with:
 *     node quiz_runner.js
 */

const readline = require("readline");

// The quiz questions, options, correct answers, and explanations are
// embedded below exactly as they were generated. Feel free to edit this
// array by hand if you want to tweak wording or add/remove questions.
const QUIZ_DATA = {quiz_json};

const rl = readline.createInterface({{
  input: process.stdin,
  output: process.stdout,
}});

// Wrap readline's callback-based question() in a Promise so we can use
// async/await and keep the flow easy to follow.
function ask(prompt) {{
  return new Promise((resolve) => rl.question(prompt, resolve));
}}

async function askQuestion(index, item) {{
  console.log(`\\nQ${{index + 1}}. ${{item.question}}`);

  // Sort option labels so they always display in a stable, readable order.
  const labels = Object.keys(item.options).sort();
  for (const label of labels) {{
    console.log(`  ${{label}}) ${{item.options[label]}}`);
  }}

  const answer = (await ask("Your answer: ")).trim().toUpperCase();
  const correct = String(item.correct).trim().toUpperCase();

  if (answer === correct) {{
    console.log("Correct!");
    return true;
  }} else {{
    console.log(`Incorrect. The correct answer was ${{item.correct}}.`);
  }}

  if (item.explanation) {{
    console.log(`Explanation: ${{item.explanation}}`);
  }}

  return false;
}}

// Simple Fisher-Yates shuffle, used only if the user opts in.
function shuffle(array) {{
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }}
  return result;
}}

async function main() {{
  console.log("=".repeat(50));
  console.log("QUIZ TIME");
  console.log("=".repeat(50));
  console.log(`This quiz has ${{QUIZ_DATA.length}} question(s).`);

  const shuffleChoice = (await ask("Shuffle question order? (y/n): ")).trim().toLowerCase();
  const questions = shuffleChoice === "y" ? shuffle(QUIZ_DATA) : QUIZ_DATA;

  let score = 0;
  for (let i = 0; i < questions.length; i++) {{
    const correct = await askQuestion(i, questions[i]);
    if (correct) score++;
  }}

  console.log("\\n" + "=".repeat(50));
  console.log(`You scored ${{score}} / ${{questions.length}}`);
  console.log("=".repeat(50));

  rl.close();
}}

main();
'''


def build_standalone_script(quiz_data: List[Dict[str, Any]], language: str = "python") -> str:
    """Dispatch helper used by the Streamlit app's download button."""
    if language == "js":
        return build_js_script(quiz_data)
    return build_python_script(quiz_data)
