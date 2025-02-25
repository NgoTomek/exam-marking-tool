import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import google.generativeai as genai
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
from PIL import Image

import sqlite3
from datetime import datetime
import pandas as pd


###############################################################################
#                        GLOBAL/HELPER FUNCTIONS
###############################################################################

def format_date(x):
    """Safely convert datetime to a nice string format."""
    try:
        dt = pd.to_datetime(x, errors='coerce')
        if pd.isnull(dt):
            return x
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return x


###############################################################################
#                        DATABASE CLASS
###############################################################################

class ExamDatabase:
    """Handles all database operations: storing submissions, question results, analytics, etc."""

    def __init__(self, db_path="exam_results.db"):
        self.db_path = db_path
        self.setup_database()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def setup_database(self):
        """Create the necessary tables if they don't exist yet."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Students table (if you track multiple students or want info)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE,
                    name TEXT
                )
            """)

            # Submissions table for storing each attempt
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    exam_code TEXT,
                    attempt_number INTEGER,
                    submission_date TIMESTAMP,
                    total_score REAL,
                    total_possible REAL,
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                )
            """)

            # Detailed question responses
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id INTEGER,
                    question_id TEXT,
                    student_answer TEXT,
                    points_earned REAL,
                    points_possible REAL,
                    feedback TEXT,
                    FOREIGN KEY (submission_id) REFERENCES submissions(id)
                )
            """)

    def get_latest_attempt(self, student_id):
        """Return the highest attempt number for a given student (0 if none)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COALESCE(MAX(attempt_number), 0)
                FROM submissions
                WHERE student_id = ?
            """, (student_id,))
            return cursor.fetchone()[0]

    def save_results(self, student_id, student_data, comparison_results):
        """
        Store the results of a single exam submission:
         - student_data: { "exam_code": <str>, "answers": {"1a":[...],...} }
         - comparison_results: { "1a": {"points":..., "total_possible":..., "feedback":[...]}, ... }
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1) Insert or ignore the student
            cursor.execute("""
                INSERT OR IGNORE INTO students (student_id)
                VALUES (?)
            """, (student_id,))

            # 2) Compute attempt number
            attempt_number = self.get_latest_attempt(student_id) + 1

            # 3) Compute total score across all questions
            total_score = sum(q["points"] for q in comparison_results.values())
            total_possible = sum(q["total_possible"] for q in comparison_results.values())

            # 4) Insert a new submission row
            cursor.execute("""
                INSERT INTO submissions
                (student_id, exam_code, attempt_number, submission_date, total_score, total_possible)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                student_id,
                student_data.get("exam_code", "Unknown"),
                attempt_number,
                datetime.now(),
                total_score,
                total_possible
            ))
            submission_id = cursor.lastrowid

            # 5) Insert question-level responses
            for question_id, result in comparison_results.items():
                # student's raw answers for that question
                raw_ans_list = student_data["answers"].get(question_id, [])
                student_answer_str = json.dumps(raw_ans_list)

                feedback_str = json.dumps(result.get("feedback", []))

                cursor.execute("""
                    INSERT INTO question_responses (
                        submission_id,
                        question_id,
                        student_answer,
                        points_earned,
                        points_possible,
                        feedback
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    submission_id,
                    question_id,
                    student_answer_str,
                    result["points"],
                    result["total_possible"],
                    feedback_str
                ))
            conn.commit()

    def get_progress_analytics(self, student_id):
        """
        Return a dict with:
          - attempts: list of (attempt_number, total_score, total_possible, percentage, submission_date)
          - question_progress: (attempt_number, question_id, points_earned, points_possible)
        """
        with self.get_connection() as conn:
            c = conn.cursor()

            # overall progress per attempt
            c.execute("""
                SELECT attempt_number, total_score, total_possible,
                       (CASE WHEN total_possible>0 THEN (total_score*100.0/total_possible) ELSE 0 END) as percentage,
                       submission_date
                FROM submissions
                WHERE student_id = ?
                ORDER BY attempt_number
            """, (student_id,))
            attempts = c.fetchall()

            # question-level progress
            c.execute("""
                SELECT s.attempt_number, qr.question_id, qr.points_earned, qr.points_possible
                FROM submissions s
                JOIN question_responses qr ON s.id = qr.submission_id
                WHERE s.student_id = ?
                ORDER BY s.attempt_number, qr.question_id
            """, (student_id,))
            question_progress = c.fetchall()

            return {
                "attempts": attempts,
                "question_progress": question_progress
            }

    def get_analytics(self):
        """
        Return overall statistics and per-question statistics
        e.g. total submissions, average score, question usage
        """
        with self.get_connection() as conn:
            c = conn.cursor()

            # overall stats
            c.execute("""
                SELECT
                    COUNT(*) as total_submissions,
                    AVG(total_score) as avg_score,
                    AVG( CASE WHEN total_possible>0 THEN (total_score*100.0/total_possible) END ) as avg_percentage
                FROM submissions
            """)
            overall = c.fetchone()

            c.execute("""
                SELECT
                    question_id,
                    COUNT(*) as attempts,
                    AVG(points_earned) as avg_score,
                    AVG( CASE WHEN points_possible>0 THEN (points_earned*100.0/points_possible) END ) as avg_percentage
                FROM question_responses
                GROUP BY question_id
                ORDER BY question_id
            """)
            question_rows = c.fetchall()

            return {
                "overall": {
                    "total_submissions": overall[0],
                    "avg_score": overall[1],
                    "avg_percentage": overall[2]
                },
                "per_question": [{
                    "question_id": r[0],
                    "attempts": r[1],
                    "avg_score": r[2],
                    "avg_percentage": r[3]
                } for r in question_rows]
            }


###############################################################################
#                   VISUALIZING & DASHBOARD
###############################################################################
def display_analytics_dashboard():
    """Show an analytics page with overall stats and per-question stats."""
    st.title("Exam Analytics Dashboard")

    db = ExamDatabase()
    analytics = db.get_analytics()

    # overall stats
    st.header("Overall Statistics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Submissions", analytics["overall"]["total_submissions"])
    if analytics["overall"]["avg_score"] is not None:
        col2.metric("Average Score", f"{analytics['overall']['avg_score']:.2f}")
    else:
        col2.metric("Average Score", "N/A")

    if analytics["overall"]["avg_percentage"] is not None:
        col3.metric("Average Percentage", f"{analytics['overall']['avg_percentage']:.1f}%")
    else:
        col3.metric("Average Percentage", "N/A")

    # question analysis
    st.header("Question-Level Analysis")
    perq = analytics["per_question"]
    if perq:
        df = pd.DataFrame(perq)
        # bar chart for average score
        st.subheader("Average Score per Question")
        st.bar_chart(df.set_index("question_id")["avg_score"])
        st.subheader("Detailed Table")
        st.dataframe(df.style.format({
            "avg_score": "{:.2f}",
            "avg_percentage": "{:.1f}%"
        }))
    else:
        st.info("No question-level data found.")


def display_student_progress(student_id):
    """Display multi-attempt progress for a single student."""
    st.title("Student Progress Analysis")

    db = ExamDatabase()
    progress_data = db.get_progress_analytics(student_id)

    # attempts table
    if not progress_data["attempts"]:
        st.warning("No attempts found for this student.")
        return

    st.header("Overall Progress")

    attempts_df = pd.DataFrame(progress_data["attempts"],
                               columns=["Attempt", "Score", "Possible", "Percentage", "Date"])
    # Convert Date to datetime
    attempts_df["Date"] = pd.to_datetime(attempts_df["Date"], errors="coerce")

    st.line_chart(attempts_df.set_index("Attempt")["Percentage"])

    st.subheader("Attempts Table")
    st.dataframe(attempts_df.style.format({
        "Score": "{:.1f}",
        "Possible": "{:.1f}",
        "Percentage": "{:.1f}%",
        "Date": format_date
    }))

    st.header("Per-Question Progress")
    q_df = pd.DataFrame(progress_data["question_progress"],
                        columns=["Attempt","Question","Points","Possible"])
    if not q_df.empty:
        pivot_df = q_df.pivot(index="Question", columns="Attempt", values="Points")
        st.dataframe(pivot_df.style.background_gradient(cmap="RdYlGn"))
    else:
        st.info("No question-level data.")


###############################################################################
#                   GEMINI PROMPT SETTINGS
###############################################################################

API_KEY = "AIzaSyCKl7FvCPoNBklNf1klaImZIbcGFXuTlYY"  # Replace with your actual key
import platform
import shutil

# Automatically detect Poppler installation
if platform.system() == "Windows":
    detected_poppler = shutil.which("pdftoppm")
    if detected_poppler:
        POPPLER_PATH = detected_poppler.rsplit("\\", 1)[0]  # Extract directory path
    else:
        POPPLER_PATH = r"C:\poppler-24.02.0\Library\bin"  # Fallback path
elif platform.system() == "Darwin":  # macOS
    POPPLER_PATH = shutil.which("pdftoppm")
else:  # Linux
    POPPLER_PATH = shutil.which("pdftoppm")

print(f"Using Poppler path: {POPPLER_PATH}")




genai.configure(api_key=API_KEY)

CONFIGS = {
    "student": {
        "prompt": """
        Analyze this image containing student exam responses and supporting work.
        Return a JSON object with the exact format:
        {
            "exam_code": "string",
            "answers": {
                "question_identifier": ["response part 1", "response part 2", ...]
            }
        }
        Only JSON, no extra text. 
        """,
        "generation_config": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 8192
        }
    },
    "markscheme": {
        "prompt": """
        Analyze this mark scheme for an exam. 
        Return a JSON object with exact format:
        {
            "exam_code": "string",
            "marking": {
                "question_identifier": [
                    {
                        "points": 1,
                        "criteria": "string",
                        "answer": "string",
                        "mark_type": "string"
                    }
                ]
            }
        }
        Where mark_type is one of "A1","B1","M1". 
        Split multi-mark lines into multiple entries. 
        Only JSON, no extra text.
        """,
        "generation_config": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 8192
        }
    }
}


###############################################################################
#                  FILE PROCESSING
###############################################################################

def process_document(file):
    """Read a single file (PDF or Image) -> list of PIL images."""
    file_bytes = file.read()
    file.seek(0)

    # check if PDF
    if file.type == "application/pdf":
        try:
            pages = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH, dpi=200)
        except Exception as e:
            st.error(f"PDF conversion error: {str(e)}")



def create_gemini_content(images, mode, prompt_text=None):
    """
    Build content dict for generate_content(...) calls.
    For each image, we pass an 'inline_data' with top-level keys 'mime_type' and raw base64 data.
    """
    if not prompt_text:
        prompt_text = CONFIGS[mode]["prompt"]

    # The new google.generativeai library expects 'parts' or 'contents' with either text or inline_data
    parts = [{"text": prompt_text}]

    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        raw = buffer.getvalue()
        # no prefix
        encoded = base64.b64encode(raw).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": encoded
            }
        })

    return {"contents": parts}


def get_gemini_response(content, mode):
    """Call Gemini with the built content and parse JSON."""
    try:
        conf = CONFIGS[mode]["generation_config"]
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite",
            generation_config=conf
        )
        resp = model.generate_content(**content)

        # Clean up any markdown fences
        text = resp.text.strip()
        if text.startswith("```json"):
            text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        # try to parse
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            st.error(f"JSON parse error: {e}")
            st.write("Raw text:")
            st.code(resp.text)
            return None

    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None


def compare_answers(student_data, markscheme_data):
    """
    Send an additional prompt to GPT with the student's answers & official marking,
    returning a structure that you can interpret for awarding points.
    For simplicity, we'll do a basic approach or skip advanced partial marks here.
    """
    student_answers = student_data.get("answers", {})
    marking = markscheme_data.get("marking", {})

    # We'll build a new prompt to let Gemini evaluate.
    compare_prompt = f"""
    You are an examiner. Compare the following student answers with the official marking.
    Return JSON of the form:
    {{
      "evaluations": {{
        "question_id": {{
           "points": <float>,
           "total_possible": <float>,
           "feedback": [
             {{
               "criterion": "string",
               "points_awarded": <float>,
               "max_points": <float>,
               "comment": "string"
             }}
           ]
        }},
        ...
      }}
    }}

    STUDENT ANSWERS:
    {json.dumps(student_answers, indent=2)}

    OFFICIAL MARKING:
    {json.dumps(marking, indent=2)}

    Evaluate carefully, awarding points for correct steps, partial credit for correct method, etc.
    """
    content = create_gemini_content([], "markscheme", compare_prompt)
    results = get_gemini_response(content, "markscheme")

    if results and "evaluations" in results:
        return results["evaluations"]
    else:
        return {}


def display_results(eval_results):
    """
    Show the final comparison results.
    eval_results: { "1a": {"points":..., "total_possible":..., "feedback": [...]} }
    """
    if not eval_results:
        st.warning("No evaluation results returned.")
        return
    total_points = 0
    total_possible = 0

    st.header("Detailed Comparison")
    for qid, qres in sorted(eval_results.items()):
        pts = qres.get("points", 0)
        maxp = qres.get("total_possible", 0)
        total_points += pts
        total_possible += maxp

        with st.expander(f"Question {qid} => {pts}/{maxp} marks", expanded=False):
            feedback_list = qres.get("feedback", [])
            for fb in feedback_list:
                crit = fb.get("criterion", "No criterion")
                pa = fb.get("points_awarded", 0)
                mp = fb.get("max_points", 0)
                comm = fb.get("comment", "")
                st.write(f"**{crit}**: {pa}/{mp} marks => {comm}")

    if total_possible > 0:
        pct = (total_points / total_possible) * 100
        st.subheader(f"Overall Score: {total_points:.2f}/{total_possible:.2f} ({pct:.1f}%)")
    else:
        st.info("No marks assigned in the comparison.")


###############################################################################
#                                MAIN APP
###############################################################################

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Exam Grading", "Analytics", "Student Progress"])

    if page == "Exam Grading":
        st.title("Exam Answer Comparison System")

        # Student ID
        student_id = st.text_input("Student ID (e.g. 'stu001')", key="student_id")
        if not student_id:
            st.warning("Enter a Student ID to proceed.")
            return

        # Student Answers
        st.subheader("Upload Student Answers")
        student_files = st.file_uploader("PDF/Image for Student Answers",
                                         type=["pdf","png","jpg","jpeg"],
                                         accept_multiple_files=True)

        # Mark Scheme
        st.subheader("Upload Mark Scheme")
        scheme_file = st.file_uploader("PDF/Image for Mark Scheme",
                                       type=["pdf","png","jpg","jpeg"])

        # Compare button
        if st.button("Process & Compare", disabled=not (student_files and scheme_file)):
            # 1) Convert student files => images => Gemini
            st.write("Processing Student Answers...")
            all_student_imgs = []
            for sf in student_files:
                pimgs = process_document(sf)
                all_student_imgs.extend(pimgs)

            if not all_student_imgs:
                st.error("No student images found or failed conversion.")
                return

            student_content = create_gemini_content(all_student_imgs, "student")
            student_data = get_gemini_response(student_content, "student")
            if not student_data:
                st.error("Failed to parse student data from Gemini.")
                return

            # 2) Convert mark scheme => images => Gemini
            st.write("Processing Mark Scheme...")
            scheme_imgs = process_document(scheme_file)
            if not scheme_imgs:
                st.error("No mark scheme images found or failed conversion.")
                return

            scheme_content = create_gemini_content(scheme_imgs, "markscheme")
            scheme_data = get_gemini_response(scheme_content, "markscheme")
            if not scheme_data:
                st.error("Failed to parse mark scheme data from Gemini.")
                return

            # 3) Compare
            st.write("Comparing Student vs Mark Scheme...")
            comparison = compare_answers(student_data, scheme_data)

            # 4) Display
            display_results(comparison)

            # 5) Save to DB
            db = ExamDatabase()
            db.save_results(student_id, student_data, comparison)

            # Show prior attempts
            with st.expander("Previous Attempts for this Student"):
                prog = db.get_progress_analytics(student_id)
                if prog["attempts"]:
                    attempts_df = pd.DataFrame(prog["attempts"],
                                               columns=["Attempt","Score","Possible","Percentage","Date"])
                    st.dataframe(attempts_df.style.format({
                        "Score":"{:.1f}",
                        "Possible":"{:.1f}",
                        "Percentage":"{:.1f}%",
                        "Date":format_date
                    }))
                else:
                    st.info("No previous attempts")

            # Show raw data for debugging
            with st.expander("Show Raw Data"):
                st.subheader("Student Data")
                st.json(student_data)
                st.subheader("Mark Scheme Data")
                st.json(scheme_data)
                st.subheader("Comparison Result")
                st.json(comparison)


    elif page == "Analytics":
        display_analytics_dashboard()

    else:  # Student Progress
        st.title("Student Progress View")
        sid = st.text_input("Enter a Student ID:")
        if sid:
            display_student_progress(sid)


if __name__ == "__main__":
    main()
