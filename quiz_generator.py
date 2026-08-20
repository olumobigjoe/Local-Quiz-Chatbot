"""
quiz_generator.py
------------------
Core logic for turning extracted document text into a multiple-choice quiz
using a locally running Ollama server (model: qwen3:4b by default).

Everything here is written defensively because a small 4B model will
sometimes:
  - wrap its JSON answer in ```json fences
  - add a sentence of preamble before/after the JSON
  - drift slightly from the requested option count
So we validate and repair aggressively rather than trusting the raw output.
"""

import json
import os
import re
import requests
from typing import List, Dict, Any, Tuple

from file_parser import chunk_text

# --- Configuration (override via environment variables / .env) -----------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT", "180"))
CHUNK_WORD_SIZE = int(os.environ.get("CHUNK_WORD_SIZE", "900"))
# How many questions to ask for from a single chunk in one call. Keeping
# this modest (rather than asking for all remaining questions at once)
# gives the small model a better shot at valid, well-formed JSON.
MAX_QUESTIONS_PER_CALL = 5


def is_ollama_reachable(host: str = OLLAMA_HOST) -> bool:
    """Quick health check so the UI can show a clear error instead of a
    confusing connection-refused traceback."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _int_to_roman(num: int) -> str:
    """Convert a small positive integer (1-20 is plenty for quiz options)
    to an uppercase Roman numeral string."""
    value_map = [
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    result = ""
    for value, symbol in value_map:
        while num >= value:
            result += symbol
            num -= value
    return result


def label_for_index(index: int, style: str) -> str:
    """
    Return the option label for a given zero-based index.
    style: 'alphabetic' -> A, B, C, ...
           'roman'      -> I, II, III, ...
    """
    if style == "roman":
        return _int_to_roman(index + 1)
    # Default / alphabetic: A-Z, then AA, AB... (unlikely to be needed but
    # avoids a crash if someone asks for a huge number of options).
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < 26:
        return letters[index]
    return letters[index // 26 - 1] + letters[index % 26]


def build_prompt(chunk: str, num_questions: int, num_options: int, label_style: str) -> str:
    """
    Build the instruction prompt sent to qwen3:4b. We ask explicitly for
    ONLY JSON, with a strict schema, and give a worked example so the small
    model has a concrete pattern to imitate.
    """
    labels = [label_for_index(i, label_style) for i in range(num_options)]
    example_options = {lbl: f"Option {lbl} text" for lbl in labels}
    example = {
        "question": "Example question text based on the source material?",
        "options": example_options,
        "correct": labels[0],
        "explanation": "One short sentence on why this answer is correct.",
    }

    return f"""You are a quiz-writing assistant. Using ONLY the SOURCE TEXT below,
write exactly {num_questions} multiple-choice question(s), each with exactly
{num_options} answer options labeled: {", ".join(labels)}.

Rules:
- Base every question strictly on facts stated in the SOURCE TEXT.
- Exactly one option must be correct.
- Keep each question and option concise (under 30 words).
- Respond with ONLY a JSON array, no prose, no markdown code fences, no explanation
  outside the JSON.
- Follow this exact schema for each item:
{json.dumps(example, indent=2)}

SOURCE TEXT:
\"\"\"
{chunk}
\"\"\"

Return the JSON array now:"""


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if the model added them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_array(text: str) -> str:
    """
    As a last resort, pull out the substring between the first '[' and the
    last ']' — handles cases where the model added a sentence of preamble
    or trailing commentary around otherwise-valid JSON.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _parse_questions(raw_text: str) -> List[Dict[str, Any]]:
    """Try progressively looser strategies to turn model output into a
    list of question dicts. Returns [] if all strategies fail."""
    candidates = [raw_text, _strip_json_fences(raw_text)]
    candidates.append(_extract_json_array(candidates[-1]))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Some models wrap the array in a top-level object.
                for value in data.values():
                    if isinstance(value, list):
                        return value
        except (json.JSONDecodeError, TypeError):
            continue
    return []


def _is_valid_question(item: Dict[str, Any], num_options: int) -> bool:
    """Sanity-check a single parsed question before we accept it."""
    if not isinstance(item, dict):
        return False
    if "question" not in item or "options" not in item or "correct" not in item:
        return False
    if not isinstance(item["options"], dict):
        return False
    if len(item["options"]) != num_options:
        return False
    if item["correct"] not in item["options"]:
        return False
    return True


def call_ollama(prompt: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
                 temperature: float = 0.3) -> str:
    """
    Send a single-turn generation request to the local Ollama server.
    stream=False keeps this simple: we wait for the full response instead
    of handling a streaming connection, which is fine for a batch quiz job.
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def generate_quiz_from_text(
    full_text: str,
    total_questions: int,
    num_options: int,
    label_style: str,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Orchestrates the whole generation flow:
      1. Chunk the source text.
      2. Walk through chunks, asking for a handful of questions at a time.
      3. Validate/repair each response; keep only well-formed questions.
      4. Stop once we have enough questions or run out of chunks.

    Returns (quiz_list, warnings) where warnings is a list of human-readable
    strings (e.g. "only generated 6/10 questions") to surface in the UI.
    """
    warnings: List[str] = []
    chunks = chunk_text(full_text, CHUNK_WORD_SIZE)

    collected: List[Dict[str, Any]] = []
    chunk_index = 0

    while len(collected) < total_questions and chunk_index < len(chunks):
        remaining = total_questions - len(collected)
        request_count = min(remaining, MAX_QUESTIONS_PER_CALL)
        chunk = chunks[chunk_index]

        prompt = build_prompt(chunk, request_count, num_options, label_style)

        # Try once, and if parsing fails completely, retry the same chunk
        # once more before giving up on it — small models occasionally
        # produce malformed output on the first attempt but succeed on a
        # second try with the same prompt.
        items: List[Dict[str, Any]] = []
        for attempt in range(2):
            try:
                raw = call_ollama(prompt, model=model, host=host)
            except requests.exceptions.RequestException as exc:
                warnings.append(f"Ollama request failed on chunk {chunk_index + 1}: {exc}")
                break
            parsed = _parse_questions(raw)
            items = [q for q in parsed if _is_valid_question(q, num_options)]
            if items:
                break  # got usable output, no need to retry

        for item in items:
            if len(collected) >= total_questions:
                break
            collected.append(item)

        chunk_index += 1

    if len(collected) < total_questions:
        warnings.append(
            f"Only generated {len(collected)} of {total_questions} requested questions "
            f"(the document may be too short, or the model skipped some malformed items)."
        )

    return collected, warnings
