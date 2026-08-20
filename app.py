"""
Course Material -> Quiz Generator (local LLM via Ollama)
==========================================================
Run with:  streamlit run app.py

Upload a PDF or DOCX of course material. This tool reads the text,
sends it to a locally-running Ollama model to draft multiple-choice
questions (with plausible wrong options AND a suggested correct
answer), lets you review/edit everything, then exports a completely
separate, self-contained `quiz_app.py` + `README.md` you can hand to
students or push to GitHub. The exported quiz app needs only
`streamlit` to run -- it does NOT need Ollama, since the questions and
answers are baked in as data.

Requires Ollama running locally (https://ollama.com) with at least one
model pulled, e.g.:
    ollama pull llama3.2:1b
    ollama pull qwen3:4b
"""

import base64
import io
import json
import re
import zipfile
from datetime import datetime

import requests
import streamlit as st

# ============================================================================
# CONFIG
# ============================================================================
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
MODEL_OPTIONS = ["llama3.2:1b", "qwen3:4b"]
CHUNK_CHARS = 3000  # ~ safe chunk size for small local models

st.set_page_config(page_title="Quiz Generator (Local LLM)", page_icon="📚", layout="wide")

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    "raw_text": "",
    "source_filename": "",
    "questions": [],
    "generation_log": [],
    "generated_once": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# FILE EXTRACTION
# ============================================================================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    import docx

    doc = docx.Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return "\n".join(parts)


def chunk_text(text: str, chunk_size: int = CHUNK_CHARS):
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # try to break on a paragraph/sentence boundary, not mid-word
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size * 0.5:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.3:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


# ============================================================================
# OLLAMA CALL + PARSING
# ============================================================================
PROMPT_TEMPLATE = """You are helping a university instructor create multiple-choice quiz questions for students, based ONLY on the course material provided below.

Generate exactly {n} multiple-choice questions from this material.
Each question must have exactly {opts} answer options.
Exactly one option must be the correct answer; the rest must be plausible but clearly wrong to someone who understood the material.
Vary which option position (0, 1, 2, ...) holds the correct answer -- do not always put it first.

Return ONLY a valid JSON array, with no text before or after it. Each element must have exactly this shape:
{{
  "question": "the question text",
  "options": ["option A text", "option B text", "..."],
  "correct_index": 0,
  "explanation": "one short sentence explaining why that answer is correct"
}}

"correct_index" is the 0-based index into "options" of the correct answer.

COURSE MATERIAL:
\"\"\"
{material}
\"\"\"
"""


def call_ollama(model: str, prompt: str, host: str) -> str:
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.4},
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def extract_json_array(raw: str):
    """Best-effort extraction of a JSON array from a model response,
    even if the model added stray text around it."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def validate_questions(items, num_options: int):
    valid = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        opts = item.get("options")
        idx = item.get("correct_index")
        if not isinstance(q, str) or not q.strip():
            continue
        if not isinstance(opts, list) or len(opts) != num_options:
            continue
        if not all(isinstance(o, str) and o.strip() for o in opts):
            continue
        if not isinstance(idx, int) or not (0 <= idx < num_options):
            continue
        valid.append(
            {
                "question": q.strip(),
                "options": [o.strip() for o in opts],
                "correct_index": idx,
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )
    return valid


def generate_quiz(text: str, model: str, host: str, total_questions: int, num_options: int):
    chunks = chunk_text(text)
    if not chunks:
        return [], ["No text could be extracted from the document."]

    # distribute questions across chunks proportionally
    n_chunks = len(chunks)
    base = total_questions // n_chunks
    remainder = total_questions % n_chunks
    per_chunk = [base + (1 if i < remainder else 0) for i in range(n_chunks)]

    all_questions = []
    log = []
    progress = st.progress(0.0, text="Starting generation...")
    for i, (chunk, n) in enumerate(zip(chunks, per_chunk)):
        progress.progress((i) / n_chunks, text=f"Generating from section {i + 1}/{n_chunks}...")
        if n <= 0:
            continue
        prompt = PROMPT_TEMPLATE.format(n=n, opts=num_options, material=chunk)
        try:
            raw = call_ollama(model, prompt, host)
        except requests.exceptions.ConnectionError:
            log.append(
                f"Section {i + 1}: could not connect to Ollama at {host}. "
                "Is `ollama serve` running?"
            )
            continue
        except Exception as e:  # noqa: BLE001
            log.append(f"Section {i + 1}: request failed ({e}).")
            continue

        parsed = extract_json_array(raw)
        valid = validate_questions(parsed, num_options)
        if not valid:
            log.append(f"Section {i + 1}: model did not return usable questions, skipped.")
        else:
            all_questions.extend(valid)
            log.append(f"Section {i + 1}: generated {len(valid)} valid question(s).")

    progress.progress(1.0, text="Done.")
    return all_questions[:total_questions], log


# ============================================================================
# EXPORT TEMPLATE (the standalone student-facing quiz app)
# ============================================================================
QUIZ_APP_TEMPLATE = r'''"""
__QUIZ_TITLE__
Auto-generated quiz app -- self-contained, no external services needed.
Generated on __GENERATED_DATE__ from: __SOURCE_FILENAME__

Run with:
    pip install streamlit
    streamlit run quiz_app.py
"""

import base64
import json

import streamlit as st

QUESTIONS_B64 = "__QUESTIONS_B64__"
QUESTIONS = json.loads(base64.b64decode(QUESTIONS_B64).decode("utf-8"))

st.set_page_config(page_title="__QUIZ_TITLE__", page_icon="\U0001F4DA", layout="centered")

st.title("__QUIZ_TITLE__")
st.caption(f"{len(QUESTIONS)} questions | Generated from: __SOURCE_FILENAME__")
st.write("")

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

with st.form("quiz_form"):
    answers = {}
    for i, q in enumerate(QUESTIONS):
        st.markdown(f"**{i + 1}. {q['question']}**")
        answers[i] = st.radio(
            label=f"q_{i}",
            options=list(range(len(q["options"]))),
            format_func=lambda idx, opts=q["options"]: opts[idx],
            key=f"answer_{i}",
            label_visibility="collapsed",
            index=None,
        )
        st.write("")
    submitted = st.form_submit_button("Submit Quiz", use_container_width=True)

if submitted:
    st.session_state.quiz_submitted = True

if st.session_state.quiz_submitted:
    unanswered = [i for i, a in answers.items() if a is None]
    if unanswered:
        st.warning(
            f"You left {len(unanswered)} question(s) unanswered: "
            f"{', '.join(str(i + 1) for i in unanswered)}. "
            "They're marked wrong below -- answer everything and resubmit to update your score."
        )

    score = sum(1 for i, q in enumerate(QUESTIONS) if answers.get(i) == q["correct_index"])
    total = len(QUESTIONS)
    pct = round(100 * score / total) if total else 0
    st.header(f"Score: {score} / {total} ({pct}%)")
    st.write("")
    st.subheader("Review")

    for i, q in enumerate(QUESTIONS):
        user_idx = answers.get(i)
        correct_idx = q["correct_index"]
        is_correct = user_idx == correct_idx
        icon = "\u2705" if is_correct else "\u274C"
        with st.expander(f"{icon} {i + 1}. {q['question']}"):
            for j, opt in enumerate(q["options"]):
                prefix = ""
                if j == correct_idx:
                    prefix = "\u2705 "
                elif j == user_idx:
                    prefix = "\u274C "
                st.write(f"{prefix}{opt}")
            if q.get("explanation"):
                st.caption(f"Why: {q['explanation']}")

    st.write("")
    if st.button("Restart Quiz"):
        for i in range(len(QUESTIONS)):
            st.session_state.pop(f"answer_{i}", None)
        st.session_state.quiz_submitted = False
        st.rerun()
'''

README_TEMPLATE = r"""# __QUIZ_TITLE__

An auto-generated, self-contained multiple-choice quiz app.

- **__NUM_QUESTIONS__ questions**, __NUM_OPTIONS__ options each
- Generated on __GENERATED_DATE__ from: `__SOURCE_FILENAME__`
- Questions were AI-drafted from the course material using a local model
  (__MODEL_USED__) and reviewed/edited before export.
- **No Ollama or internet connection needed to run this quiz** -- the
  questions, options, and answers are embedded directly in `quiz_app.py`.

## Run it

```bash
pip install streamlit
streamlit run quiz_app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Deploy it (optional)

Push `quiz_app.py` (and this README) to a GitHub repo, then deploy on
[Streamlit Community Cloud](https://share.streamlit.io) with the main
file path set to `quiz_app.py` -- no other setup or secrets required.

## Notes

- AI-drafted answers were reviewed by the instructor before export, but
  always spot-check a generated quiz before giving it to students.
- To regenerate or create a different quiz from new material, use the
  quiz builder tool (`app.py`) again -- this exported file is a
  standalone snapshot and does not need it to run.
"""


def build_export_files(questions, source_filename, model_used):
    quiz_title = f"Quiz: {source_filename.rsplit('.', 1)[0]}" if source_filename else "Course Quiz"
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    questions_json = json.dumps(questions, ensure_ascii=False)
    questions_b64 = base64.b64encode(questions_json.encode("utf-8")).decode("ascii")

    quiz_code = (
        QUIZ_APP_TEMPLATE.replace("__QUIZ_TITLE__", quiz_title)
        .replace("__GENERATED_DATE__", generated_date)
        .replace("__SOURCE_FILENAME__", source_filename or "unknown")
        .replace("__QUESTIONS_B64__", questions_b64)
    )
    readme = (
        README_TEMPLATE.replace("__QUIZ_TITLE__", quiz_title)
        .replace("__NUM_QUESTIONS__", str(len(questions)))
        .replace("__NUM_OPTIONS__", str(len(questions[0]["options"])) if questions else "?")
        .replace("__GENERATED_DATE__", generated_date)
        .replace("__SOURCE_FILENAME__", source_filename or "unknown")
        .replace("__MODEL_USED__", model_used)
    )
    return quiz_code, readme


# ============================================================================
# UI -- chat-style walkthrough
# ============================================================================
st.title("📚 Course Material → Quiz Generator")
st.caption("Runs entirely on your machine via Ollama. No data leaves your laptop.")

with st.chat_message("assistant"):
    st.write(
        "Upload a PDF or DOCX of your course material, choose how many "
        "questions you want, and I'll draft a multiple-choice quiz from "
        "it -- including suggested correct answers -- for you to review."
    )

# ---- Sidebar settings ----
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Local model",
        MODEL_OPTIONS,
        index=0,
        help="llama3.2:1b is faster and lighter (recommended default on 8GB RAM). "
        "qwen3:4b tends to write better questions but is slower on CPU.",
    )
    ollama_host = st.text_input("Ollama host", value=DEFAULT_OLLAMA_HOST)
    num_questions = st.slider("Number of questions", 3, 30, 10)
    num_options = st.slider("Options per question", 2, 6, 4)
    st.divider()
    st.caption(
        "Make sure Ollama is running (`ollama serve`) and the model is "
        f"pulled (`ollama pull {model}`)."
    )

# ---- File upload ----
uploaded = st.file_uploader("Upload course material", type=["pdf", "docx"])

if uploaded is not None and uploaded.name != st.session_state.source_filename:
    with st.spinner(f"Reading {uploaded.name}..."):
        file_bytes = uploaded.read()
        if uploaded.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        else:
            text = extract_text_from_docx(file_bytes)
    st.session_state.raw_text = text
    st.session_state.source_filename = uploaded.name
    st.session_state.questions = []
    st.session_state.generated_once = False

if st.session_state.raw_text:
    word_count = len(st.session_state.raw_text.split())
    with st.chat_message("assistant"):
        st.write(
            f"I read **{word_count:,} words** from `{st.session_state.source_filename}`. "
            f"Ready to draft **{num_questions} questions** ({num_options} options each) "
            f"using **{model}**."
        )
    if word_count < 30:
        st.warning(
            "That's very little extracted text -- if this is a scanned/image-only "
            "PDF, text extraction won't work. Try a text-based PDF or DOCX."
        )

    if st.button("🚀 Generate Quiz", type="primary", disabled=word_count < 30):
        with st.spinner("Talking to the local model... this can take a while on CPU."):
            questions, log = generate_quiz(
                st.session_state.raw_text, model, ollama_host, num_questions, num_options
            )
        st.session_state.questions = questions
        st.session_state.generation_log = log
        st.session_state.generated_once = True

# ---- Generation log / errors ----
if st.session_state.generation_log:
    with st.expander("Generation log", expanded=not st.session_state.questions):
        for line in st.session_state.generation_log:
            st.write("- " + line)

# ---- Review & edit ----
if st.session_state.questions:
    with st.chat_message("assistant"):
        st.write(
            f"Drafted **{len(st.session_state.questions)}** question(s). "
            "Review and edit below -- fix wording, swap options, or change "
            "the marked correct answer -- then export."
        )

    to_remove = []
    for i, q in enumerate(st.session_state.questions):
        with st.expander(f"Q{i + 1}: {q['question'][:80]}", expanded=False):
            q["question"] = st.text_area("Question", value=q["question"], key=f"qtext_{i}")
            for j in range(len(q["options"])):
                q["options"][j] = st.text_input(
                    f"Option {j + 1}", value=q["options"][j], key=f"opt_{i}_{j}"
                )
            q["correct_index"] = st.radio(
                "Correct answer",
                options=list(range(len(q["options"]))),
                format_func=lambda idx, opts=q["options"]: opts[idx],
                index=q["correct_index"],
                key=f"correct_{i}",
            )
            q["explanation"] = st.text_input(
                "Explanation (optional)", value=q.get("explanation", ""), key=f"expl_{i}"
            )
            if st.checkbox("Remove this question", key=f"remove_{i}"):
                to_remove.append(i)

    if to_remove:
        st.session_state.questions = [
            q for i, q in enumerate(st.session_state.questions) if i not in to_remove
        ]
        st.rerun()

    st.divider()
    st.subheader("Export")
    if st.session_state.questions:
        quiz_code, readme = build_export_files(
            st.session_state.questions, st.session_state.source_filename, model
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "⬇️ Download quiz_app.py",
                data=quiz_code,
                file_name="quiz_app.py",
                mime="text/x-python",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇️ Download README.md",
                data=readme,
                file_name="README.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col3:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("quiz_app.py", quiz_code)
                zf.writestr("README.md", readme)
            st.download_button(
                "⬇️ Download both (.zip)",
                data=zip_buf.getvalue(),
                file_name="generated_quiz.zip",
                mime="application/zip",
                use_container_width=True,
            )

        with st.expander("Preview generated quiz_app.py"):
            st.code(quiz_code, language="python")
    else:
        st.info("All questions were removed -- nothing to export.")
