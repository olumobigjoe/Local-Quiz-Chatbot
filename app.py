"""
app.py
-------
Streamlit UI for the local quiz-generator chatbot.

Flow:
  1. User uploads a .docx or .pdf file.
  2. User sets number of questions, number of options, and label style.
  3. On click, we extract text, chunk it, and call local Ollama (qwen3:4b)
     to generate the quiz.
  4. The quiz is shown in the app, and download buttons let the user save:
       - the quiz as a standalone Python or JS script
       - a README.md
       - a requirements.txt
       - the raw quiz as JSON

Run with:
    streamlit run app.py
"""

import json
import os
import tempfile

import streamlit as st

from file_parser import extract_text
from quiz_generator import generate_quiz_from_text, is_ollama_reachable, OLLAMA_HOST, OLLAMA_MODEL
from standalone_script_builder import build_standalone_script
from project_files import README_CONTENT, REQUIREMENTS_TXT, PACKAGE_JSON

st.set_page_config(page_title="Local Quiz Generator (Ollama)", page_icon="📝", layout="centered")

st.title("📝 Local Quiz Generator")
st.caption(f"Powered by your local Ollama model — running fully offline.")

# --- Session state -----------------------------------------------------
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "warnings" not in st.session_state:
    st.session_state.warnings = []

# --- Sidebar: connection + generation settings --------------------------
with st.sidebar:
    st.header("Settings")

    ollama_host = st.text_input("Ollama host", value=OLLAMA_HOST)
    ollama_model = st.text_input("Model name", value=OLLAMA_MODEL)

    st.divider()

    num_questions = st.number_input("Number of questions", min_value=1, max_value=50, value=5, step=1)
    num_options = st.number_input("Options per question", min_value=2, max_value=6, value=4, step=1)
    label_style = st.radio("Option labels", options=["alphabetic", "roman"],
                            format_func=lambda s: "Alphabetic (A, B, C...)" if s == "alphabetic" else "Roman numeral (I, II, III...)")

    st.divider()

    output_language = st.radio("Downloadable quiz script language", options=["python", "js"],
                                format_func=lambda s: "Python" if s == "python" else "JavaScript")

# --- Main: file upload ---------------------------------------------------
uploaded_file = st.file_uploader("Upload a Word (.docx) or PDF (.pdf) document", type=["docx", "pdf"])

generate_clicked = st.button("Generate Quiz", type="primary", disabled=uploaded_file is None)

if generate_clicked and uploaded_file is not None:
    # Health-check Ollama first so we can show a clear, actionable error
    # instead of a raw connection traceback.
    if not is_ollama_reachable(ollama_host):
        st.error(
            f"Could not reach Ollama at {ollama_host}. "
            f"Make sure it's running locally (`ollama serve`) and that "
            f"the model is pulled (`ollama pull {ollama_model}`)."
        )
    else:
        # Streamlit's uploaded file is an in-memory buffer; write it to a
        # temp file so python-docx / pdfplumber (which expect file paths)
        # can read it.
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Reading document..."):
                text = extract_text(tmp_path)

            with st.spinner(f"Generating {num_questions} question(s) with {ollama_model}... "
                             f"this can take a while on CPU-only hardware."):
                quiz, warnings = generate_quiz_from_text(
                    full_text=text,
                    total_questions=int(num_questions),
                    num_options=int(num_options),
                    label_style=label_style,
                    model=ollama_model,
                    host=ollama_host,
                )

            st.session_state.quiz = quiz
            st.session_state.warnings = warnings

            if not quiz:
                st.error("No valid questions could be generated. Try a shorter document, "
                          "fewer questions, or check that the model is responding correctly.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error to the user
            st.error(f"Something went wrong: {exc}")
        finally:
            os.unlink(tmp_path)

# --- Display generated quiz + downloads ----------------------------------
if st.session_state.quiz:
    quiz = st.session_state.quiz

    for warning in st.session_state.warnings:
        st.warning(warning)

    st.subheader(f"Generated Quiz ({len(quiz)} question{'s' if len(quiz) != 1 else ''})")

    for i, item in enumerate(quiz, start=1):
        with st.expander(f"Q{i}. {item['question']}", expanded=True):
            for label in sorted(item["options"].keys()):
                st.write(f"**{label})** {item['options'][label]}")
            st.markdown(f"✅ **Correct answer:** {item['correct']}")
            if item.get("explanation"):
                st.caption(item["explanation"])

    st.divider()
    st.subheader("Downloads")

    col1, col2 = st.columns(2)

    with col1:
        script_content = build_standalone_script(quiz, language=output_language)
        script_filename = "quiz_runner.py" if output_language == "python" else "quiz_runner.js"
        st.download_button(
            "⬇️ Download quiz script",
            data=script_content,
            file_name=script_filename,
            mime="text/plain",
        )

        st.download_button(
            "⬇️ Download README.md",
            data=README_CONTENT,
            file_name="README.md",
            mime="text/markdown",
        )

    with col2:
        deps_content = REQUIREMENTS_TXT if output_language == "python" else PACKAGE_JSON
        deps_filename = "requirements.txt" if output_language == "python" else "package.json"
        st.download_button(
            f"⬇️ Download {deps_filename}",
            data=deps_content,
            file_name=deps_filename,
            mime="text/plain",
        )

        st.download_button(
            "⬇️ Download raw quiz (JSON)",
            data=json.dumps(quiz, indent=2),
            file_name="quiz_data.json",
            mime="application/json",
        )
else:
    st.info("Upload a document and click **Generate Quiz** to get started.")
