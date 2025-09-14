# resume_screening_app.py
# Run: pip install streamlit scikit-learn PyPDF2 python-docx pandas numpy pytesseract pdf2image pillow
# Then: streamlit run resume_screening_app.py

import io
import re
import os
import math
import base64
import json
import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Dict, Tuple
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OCR
import pytesseract
from pdf2image import convert_from_bytes

# ------------------------------- Storage Helpers -------------------------------- #

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

def list_projects():
    return [f.stem for f in PROJECTS_DIR.glob("*.json")]

def save_project(name, data):
    with open(PROJECTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_project(name):
    path = PROJECTS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def delete_project_file(name):
    path = PROJECTS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()

# ------------------------------- Text & Scoring Helpers -------------------------------- #

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b", " email ", text)
    text = re.sub(r"http[s]?://\S+", " url ", text)
    text = re.sub(r"\d+(\.\d+)?", " num ", text)
    text = re.sub(r"[^a-z0-9\-\+\#\.\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_pdf(file: io.BytesIO) -> str:
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)
    except Exception:
        return ""

def read_docx(file: io.BytesIO) -> str:
    try:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def read_txt(file: io.BytesIO) -> str:
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return file.read().decode(enc)
        except Exception:
            file.seek(0)
            continue
    return ""

def extract_text_from_file(uploaded_file) -> str:
    raw = uploaded_file.read()
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    if ext == ".pdf":
        text = read_pdf(io.BytesIO(raw))
        if not text.strip():
            # fallback to OCR
            try:
                images = convert_from_bytes(raw)
                ocr_texts = [pytesseract.image_to_string(img) for img in images]
                text = "\n".join(ocr_texts)
            except Exception as e:
                print(f"OCR failed for {uploaded_file.name}: {e}")
    elif ext in (".docx", ".doc"):
        text = read_docx(io.BytesIO(raw))
    elif ext in (".txt",):
        text = read_txt(io.BytesIO(raw))
    else:
        text = read_pdf(io.BytesIO(raw))
        if not text.strip():
            text = read_docx(io.BytesIO(raw))
        if not text.strip():
            text = read_txt(io.BytesIO(raw))
    return text

def parse_skill_list(s: str) -> List[str]:
    items = []
    for chunk in re.split(r"[,\n;]", s or ""):
        item = chunk.strip().lower()
        if item:
            items.append(item)
    seen = set()
    uniq = []
    for x in items:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def skill_matches(text: str, skills: List[str]) -> Tuple[int, List[str], List[str]]:
    matched = []
    missing = []
    for sk in skills:
        pattern = re.escape(sk)
        if re.search(rf"(?<!\w){pattern}(?!\w)", text, flags=re.IGNORECASE):
            matched.append(sk)
        else:
            missing.append(sk)
    return len(matched), matched, missing

def compute_similarity(job_text: str, resume_text: str, vec: TfidfVectorizer) -> float:
    X = vec.transform([job_text, resume_text])
    sim = cosine_similarity(X[0:1], X[1:2])[0, 0]
    if math.isnan(sim):
        return 0.0
    return float(sim)

def to_percent(x: float) -> float:
    return round(100.0 * max(0.0, min(1.0, x)), 2)

def make_download_link(df: pd.DataFrame, filename: str = "ranked_candidates.csv") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download results as CSV</a>'
    return href

def highlight_text(text: str, skills: List[str], color: str = "lightgreen") -> str:
    text_esc = text
    for sk in skills:
        pattern = re.escape(sk)
        text_esc = re.sub(
            rf"(?i)\b({pattern})\b",
            rf"<mark style='background-color:{color};'>\1</mark>",
            text_esc
        )
    return text_esc

def recompute_results(jd: str, essential: List[str], desirable: List[str], resumes_raw: Dict[str, str], weights: Dict[str, float]) -> pd.DataFrame:
    """Recompute ranking table for given data (used after candidate deletion)."""
    jd_clean = clean_text(jd)
    resumes_clean = {name: clean_text(txt or "") for name, txt in resumes_raw.items()}

    # Normalize weights to sum 1 (safe-guard)
    w_jd = float(weights.get("w_jd", 0.5))
    w_ess = float(weights.get("w_ess", 0.4))
    w_des = float(weights.get("w_des", 0.1))
    total_w = w_jd + w_ess + w_des
    if total_w == 0:
        w_jd, w_ess, w_des = 0.5, 0.4, 0.1
        total_w = 1.0
    norm = (w_jd/total_w, w_ess/total_w, w_des/total_w)

    if len(resumes_clean) == 0:
        return pd.DataFrame(columns=[
            "Candidate","JD_Match_%","Essential_Coverage_%","Desirable_Coverage_%","Total_Score_%",
            "Matched_Essential","Missing_Essential","Matched_Desirable","Missing_Desirable"
        ])

    corpus = [jd_clean] + [resumes_clean[name] for name in resumes_clean]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95)
    vectorizer.fit(corpus)

    rows = []
    for name, rtext_clean in resumes_clean.items():
        sim = compute_similarity(jd_clean, rtext_clean, vectorizer)
        ess_count, ess_matched, ess_missing = skill_matches(rtext_clean, essential)
        des_count, des_matched, des_missing = skill_matches(rtext_clean, desirable)
        ess_cov = (ess_count / max(1, len(essential))) if essential else 0.0
        des_cov = (des_count / max(1, len(desirable))) if desirable else 0.0
        score = norm[0] * sim + norm[1] * ess_cov + norm[2] * des_cov

        rows.append({
            "Candidate": name,
            "JD_Match_%": to_percent(sim),
            "Essential_Coverage_%": to_percent(ess_cov),
            "Desirable_Coverage_%": to_percent(des_cov),
            "Total_Score_%": to_percent(score),
            "Matched_Essential": ", ".join(ess_matched) if ess_matched else "",
            "Missing_Essential": ", ".join(ess_missing) if ess_missing else "",
            "Matched_Desirable": ", ".join(des_matched) if des_matched else "",
            "Missing_Desirable": ", ".join(des_missing) if des_missing else "",
        })
    df = pd.DataFrame(rows).sort_values(by="Total_Score_%", ascending=False).reset_index(drop=True)
    return df

# ------------------------------- UI ------------------------------------- #

st.set_page_config(page_title="📄 Resume Screening App", layout="wide")
st.title("📄 Resume Screening & Ranking")

# ---- Project Manager ----
st.sidebar.header("📂 Project Manager")
existing = list_projects()
project_choice = st.sidebar.selectbox("Select Project", ["-- New Project --"] + existing)

jd, essential_raw, desirable_raw, saved_resumes, saved_weights = "", "", "", {}, None

if project_choice == "-- New Project --":
    new_name = st.sidebar.text_input("Enter new project name")
    if new_name and st.sidebar.button("Create Project"):
        project_data = {
            "jd": "", "essential": [], "desirable": [],
            "resumes": {}, "results": [], "weights": {"w_jd": 0.5, "w_ess": 0.4, "w_des": 0.1}
        }
        save_project(new_name, project_data)
        st.session_state["project"] = new_name
else:
    st.session_state["project"] = project_choice
    project_data = load_project(project_choice)
    jd = project_data.get("jd", "")
    essential_raw = "\n".join(project_data.get("essential", []))
    desirable_raw = "\n".join(project_data.get("desirable", []))
    saved_resumes = project_data.get("resumes", {})
    saved_weights = project_data.get("weights", {"w_jd": 0.5, "w_ess": 0.4, "w_des": 0.1})

# --- Sidebar: Project deletion (with confirmation) ---
if "project" in st.session_state:
    current_proj = st.session_state["project"]
    if current_proj and current_proj != "-- New Project --":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗑️ Danger Zone")
        confirm_name = st.sidebar.text_input("Type project name to confirm deletion")
        delete_click = st.sidebar.button("Delete Entire Project", type="secondary")
        if delete_click:
            if confirm_name.strip() == current_proj:
                delete_project_file(current_proj)
                st.session_state.pop("project", None)
                st.success(f"Project '{current_proj}' deleted.")
                st.rerun()
            else:
                st.sidebar.error("Project name mismatch. Deletion cancelled.")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
- Create/select a project in the sidebar.
- Enter the **Job Description**, and list **Essential** and **Desirable** skills.
- Upload resumes in **PDF, DOCX, or TXT**.
- The app computes JD match (cosine similarity), essential coverage, and desirable coverage.
- Final score = weighted sum of the three. Adjust weights in the sidebar.
- You can also view JD/resume text with **highlights** for matched skills.
- Scanned PDFs are handled using **OCR**.
- Results are saved with the project for later review.
- Adding resumes to an existing project will **merge** them with previously saved ones.
- You can **delete any candidate** or the **entire project** (Danger Zone).
        """
    )

colA, colB = st.columns([2, 1])

with colA:
    jd = st.text_area("Job Description", height=220, value=jd)

    c1, c2 = st.columns(2)
    with c1:
        essential_raw = st.text_area("Essential Skills", height=160, value=essential_raw)
    with c2:
        desirable_raw = st.text_area("Desirable Skills", height=160, value=desirable_raw)

    files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

with colB:
    st.subheader("Scoring Weights")
    # Use saved weights as defaults if available
    default_w_jd = float(saved_weights["w_jd"]) if saved_weights else 0.5
    default_w_ess = float(saved_weights["w_ess"]) if saved_weights else 0.4
    default_w_des = float(saved_weights["w_des"]) if saved_weights else 0.1

    w_jd = st.slider("JD Match Weight", 0.0, 1.0, default_w_jd, 0.05)
    w_ess = st.slider("Essential Coverage Weight", 0.0, 1.0, default_w_ess, 0.05)
    w_des = st.slider("Desirable Coverage Weight", 0.0, 1.0, default_w_des, 0.05)

    total_w = w_jd + w_ess + w_des
    if total_w == 0:
        norm = (0.0, 0.0, 0.0)
    else:
        norm = (w_jd / total_w, w_ess / total_w, w_des / total_w)

    st.caption(f"Normalized Weights → JD: {norm[0]:.2f}, Essential: {norm[1]:.2f}, Desirable: {norm[2]:.2f}")

    show_details = st.checkbox("Show per-candidate skill details", value=True)

st.markdown("---")

# ------------------------------- Tabs ------------------------------------- #

tab1, tab2 = st.tabs(["🔎 Screen & Rank", "📂 View Past Results"])

with tab1:
    run = st.button("Run Screening", type="primary", disabled=not (jd and (files or saved_resumes)))

    if run:
        essential = parse_skill_list(essential_raw)
        desirable = parse_skill_list(desirable_raw)

        jd_clean = clean_text(jd)

        # Start with saved resumes (keep old)
        resumes_raw: Dict[str, str] = dict(saved_resumes)
        resumes_clean: Dict[str, str] = {name: clean_text(txt) for name, txt in saved_resumes.items()}

        # Add/overwrite with newly uploaded files
        for f in files:
            text = extract_text_from_file(f)
            resumes_raw[f.name] = text or ""
            resumes_clean[f.name] = clean_text(text or "")

        # TF-IDF vectorizer on JD + all resumes
        corpus = [jd_clean] + [resumes_clean[name] for name in resumes_clean]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95)
        vectorizer.fit(corpus)

        rows = []
        for name, rtext_clean in resumes_clean.items():
            sim = compute_similarity(jd_clean, rtext_clean, vectorizer)

            ess_count, ess_matched, ess_missing = skill_matches(rtext_clean, essential)
            des_count, des_matched, des_missing = skill_matches(rtext_clean, desirable)

            ess_cov = (ess_count / max(1, len(essential))) if essential else 0.0
            des_cov = (des_count / max(1, len(desirable))) if desirable else 0.0

            score = norm[0] * sim + norm[1] * ess_cov + norm[2] * des_cov

            rows.append({
                "Candidate": name,
                "JD_Match_%": to_percent(sim),
                "Essential_Coverage_%": to_percent(ess_cov),
                "Desirable_Coverage_%": to_percent(des_cov),
                "Total_Score_%": to_percent(score),
                "Matched_Essential": ", ".join(ess_matched) if ess_matched else "",
                "Missing_Essential": ", ".join(ess_missing) if ess_missing else "",
                "Matched_Desirable": ", ".join(des_matched) if des_matched else "",
                "Missing_Desirable": ", ".join(des_missing) if des_missing else "",
            })

        df = pd.DataFrame(rows).sort_values(by="Total_Score_%", ascending=False).reset_index(drop=True)

        st.subheader("📊 Ranked Candidates")
        st.dataframe(
            df[["Candidate", "Total_Score_%", "JD_Match_%", "Essential_Coverage_%", "Desirable_Coverage_%"]],
            use_container_width=True
        )

        st.markdown(make_download_link(df), unsafe_allow_html=True)

        # Save project state (merged resumes) + weights
        if "project" in st.session_state:
            save_project(st.session_state["project"], {
                "jd": jd,
                "essential": essential,
                "desirable": desirable,
                "resumes": resumes_raw,   # merged
                "results": df.to_dict(orient="records"),
                "weights": {"w_jd": w_jd, "w_ess": w_ess, "w_des": w_des}
            })

        if show_details:
            st.markdown("### 🔍 Details by Candidate")
            for _, r in df.iterrows():
                with st.expander(f"{r['Candidate']} — Total Score {r['Total_Score_%']}%"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("JD Match", f"{r['JD_Match_%']}%")
                    c2.metric("Essential Coverage", f"{r['Essential_Coverage_%']}%")
                    c3.metric("Desirable Coverage", f"{r['Desirable_Coverage_%']}%")

                    st.markdown("**Matched Essential:** " + (r["Matched_Essential"] or "_None_"))
                    st.markdown("**Missing Essential:** " + (r["Missing_Essential"] or "_None_"))
                    st.markdown("**Matched Desirable:** " + (r["Matched_Desirable"] or "_None_"))
                    st.markdown("**Missing Desirable:** " + (r["Missing_Desirable"] or "_None_"))

                    if st.checkbox(f"Show JD/Resume text with highlights for {r['Candidate']}"):
                        jd_highlighted = highlight_text(jd, parse_skill_list(essential_raw) + parse_skill_list(desirable_raw), "lightblue")
                        resume_highlighted = highlight_text(
                            (resumes_raw.get(r['Candidate']) or ""), 
                            parse_skill_list(essential_raw) + parse_skill_list(desirable_raw), 
                            "lightgreen"
                        )

                        st.markdown("**Job Description with highlights:**", unsafe_allow_html=True)
                        st.markdown(f"<div style='white-space: pre-wrap;'>{jd_highlighted}</div>", unsafe_allow_html=True)

                        st.markdown("**Resume with highlights:**", unsafe_allow_html=True)
                        st.markdown(f"<div style='white-space: pre-wrap;'>{resume_highlighted}</div>", unsafe_allow_html=True)

with tab2:
    if "project" in st.session_state:
        proj = st.session_state["project"]
        if proj and proj != "-- New Project --":
            saved = load_project(proj)
            if saved.get("results"):
                st.subheader(f"📊 Saved Results for {proj}")
                df_saved = pd.DataFrame(saved["results"])
                st.dataframe(
                    df_saved[["Candidate", "Total_Score_%", "JD_Match_%", "Essential_Coverage_%", "Desirable_Coverage_%"]],
                    use_container_width=True
                )
                st.markdown(make_download_link(df_saved, filename=f"{proj}_results.csv"), unsafe_allow_html=True)

                # show saved weights summary
                sw = saved.get("weights", {"w_jd": 0.5, "w_ess": 0.4, "w_des": 0.1})
                total_sw = sw["w_jd"] + sw["w_ess"] + sw["w_des"]
                if total_sw <= 0: total_sw = 1.0
                st.caption(f"Saved Weights → JD: {sw['w_jd']/total_sw:.2f}, Essential: {sw['w_ess']/total_sw:.2f}, Desirable: {sw['w_des']/total_sw:.2f}")

                # Candidate details with delete button
                if show_details:
                    st.markdown("### 🔍 Details by Candidate (Saved)")
                    for r in saved["results"]:
                        with st.expander(f"{r['Candidate']} — Total Score {r['Total_Score_%']}%"):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("JD Match", f"{r['JD_Match_%']}%")
                            c2.metric("Essential Coverage", f"{r['Essential_Coverage_%']}%")
                            c3.metric("Desirable Coverage", f"{r['Desirable_Coverage_%']}%")

                            st.markdown("**Matched Essential:** " + (r['Matched_Essential'] or "_None_"))
                            st.markdown("**Missing Essential:** " + (r['Missing_Essential'] or "_None_"))
                            st.markdown("**Matched Desirable:** " + (r['Matched_Desirable'] or "_None_"))
                            st.markdown("**Missing Desirable:** " + (r['Missing_Desirable'] or "_None_"))

                            col_del1, col_del2 = st.columns([1,4])
                            with col_del1:
                                if st.button(f"Delete Candidate", key=f"del_{r['Candidate']}"):
                                    # Remove candidate from resumes, recompute, save, refresh
                                    resumes_raw = dict(saved.get("resumes", {}))
                                    if r["Candidate"] in resumes_raw:
                                        resumes_raw.pop(r["Candidate"])
                                        # Recompute with saved data
                                        df_new = recompute_results(
                                            jd=saved.get("jd",""),
                                            essential=saved.get("essential", []),
                                            desirable=saved.get("desirable", []),
                                            resumes_raw=resumes_raw,
                                            weights=saved.get("weights", {"w_jd":0.5,"w_ess":0.4,"w_des":0.1})
                                        )
                                        save_project(proj, {
                                            "jd": saved.get("jd",""),
                                            "essential": saved.get("essential", []),
                                            "desirable": saved.get("desirable", []),
                                            "resumes": resumes_raw,
                                            "results": df_new.to_dict(orient="records"),
                                            "weights": saved.get("weights", {"w_jd":0.5,"w_ess":0.4,"w_des":0.1})
                                        })
                                        st.success(f"Candidate '{r['Candidate']}' deleted from project.")
                                        st.rerun()
                                    else:
                                        st.warning("Candidate file not found in saved resumes data.")
            else:
                st.info("No saved results found for this project yet. Run screening first.")
        else:
            st.info("Create a new project or select an existing one to view saved results.")
    else:
        st.warning("Please select or create a project in the sidebar.")
