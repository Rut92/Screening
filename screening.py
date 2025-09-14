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

# ---------- Section parsing & weights ---------- #

SECTION_HEADERS = ["experience", "work history", "employment", "projects", "education", "skills", "summary"]

def split_sections(text: str) -> Dict[str, str]:
    parts = {"experience": "", "projects": "", "skills": "", "education": "", "summary": "", "other": ""}
    lines = (text or "").splitlines()
    current = "other"
    for line in lines:
        low = line.lower().strip()
        matched_header = None
        for hdr in SECTION_HEADERS:
            if re.search(rf"\b{re.escape(hdr)}\b", low):
                matched_header = hdr
                break
        if matched_header:
            current = matched_header if matched_header in parts else "other"
        parts[current] += line + "\n"
    return parts

SECTION_WEIGHTS = {
    "experience": 2.0,
    "projects": 1.5,
    "skills": 1.0,
    "education": 0.5,
    "summary": 0.8,
    "other": 1.0,
}

# ---------- Experience boosting helpers ---------- #

YEARS_PATTERNS = [
    r"(\d+)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience\s+)?(?:in|with)?\s*{skill}",
    r"{skill}\s+for\s+(\d+)\s*\+?\s*(?:years?|yrs?)",
    r"{skill}[^.\n]{{0,40}}(\d+)\s*\+?\s*(?:years?|yrs?)",
    r"(\d+)\s*\+?\s*(?:years?|yrs?)\s+[^.\n]{{0,40}}{skill}",
]

def count_skill_frequency(text: str, skill: str) -> int:
    return len(re.findall(rf"(?i)\b{re.escape(skill)}\b", text or ""))

def detect_years_of_experience(text: str, skill: str) -> int:
    """Return the max years detected near the skill using regex heuristics."""
    t = text or ""
    max_years = 0
    for pat in YEARS_PATTERNS:
        pat_concrete = pat.format(skill=re.escape(skill))
        for m in re.finditer(rf"(?i){pat_concrete}", t):
            try:
                years = int(m.group(1))
                if years > max_years:
                    max_years = years
            except:
                continue
    return max_years

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

def make_download_text_link(text: str, filename: str = "candidate_explanations.txt") -> str:
    b64 = base64.b64encode((text or "").encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def highlight_text(text: str, skills: List[str], color: str = "lightgreen") -> str:
    text_esc = text or ""
    for sk in skills:
        pattern = re.escape(sk)
        text_esc = re.sub(
            rf"(?i)\b({pattern})\b",
            rf"<mark style='background-color:{color};'>\1</mark>",
            text_esc
        )
    return text_esc

# ---------- Explanation helper ---------- #

def build_explanation(
    candidate: str,
    row: Dict,
    resume_text: str,
    essential: List[str],
    desirable: List[str],
    freq_map: Dict[str, int],
    years_map: Dict[str, int],
    sections: Dict[str, str]
) -> str:
    exp = [f"Candidate: {candidate}"]
    exp.append(
        f"Overall Score: {row['Total_Score_%']}% (JD Match {row['JD_Match_%']}%, Essentials {row['Essential_Coverage_%']}%, Desirables {row['Desirable_Coverage_%']}%)"
    )
    if row["Matched_Essential"]:
        exp.append(f"Matched Essential Skills: {row['Matched_Essential']}")
    if row["Missing_Essential"]:
        exp.append(f"Missing Essential Skills: {row['Missing_Essential']}")
    if row["Matched_Desirable"]:
        exp.append(f"Matched Desirable Skills: {row['Matched_Desirable']}")
    if row["Missing_Desirable"]:
        exp.append(f"Missing Desirable Skills: {row['Missing_Desirable']}")

    # Experience boost details
    boost_bits = []
    for sk in sorted(freq_map.keys()):
        parts = [f"{sk}: {freq_map[sk]} mentions"]
        yrs = years_map.get(sk, 0)
        if yrs > 0:
            parts.append(f"{yrs} years")
        boost_bits.append(" (" .join(parts) + (")" if parts else ""))
    if boost_bits:
        exp.append("Experience Evidence: " + "; ".join(boost_bits))

    # Section weighting evidence
    weighted_hits = []
    for sec, txt in sections.items():
        hits = [sk for sk in essential + desirable if re.search(rf"(?i)\b{re.escape(sk)}\b", txt or "")]
        if hits:
            weighted_hits.append(f"{sec.title()}({SECTION_WEIGHTS.get(sec, 1.0)}x): {', '.join(hits)}")
    if weighted_hits:
        exp.append("Section Weighting Evidence: " + " | ".join(weighted_hits))

    return "\n".join(exp)

# ---------- Recompute (used by delete + saved results rebuild) ---------- #

def recompute_results(
    jd: str,
    essential: List[str],
    desirable: List[str],
    resumes_raw: Dict[str, str],
    weights: Dict[str, float]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Recompute ranking table + explanations for given data (used after candidate deletion).
    Keeps your original weight normalization behavior and adds:
      - Experience boosting (frequency + 'X years' detection)
      - Section weighting
      - Explanations
    """
    jd_clean = clean_text(jd)
    resumes_clean = {name: clean_text(txt or "") for name, txt in resumes_raw.items()}

    # Normalize weights to sum 1 (same behavior you had in the main view)
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
        ]), []

    corpus = [jd_clean] + [resumes_clean[name] for name in resumes_clean]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95)
    vectorizer.fit(corpus)

    rows = []
    explanations = []

    for name, rtext_clean in resumes_clean.items():
        sim = compute_similarity(jd_clean, rtext_clean, vectorizer)

        ess_count, ess_matched, ess_missing = skill_matches(rtext_clean, essential)
        des_count, des_matched, des_missing = skill_matches(rtext_clean, desirable)

        ess_cov = (ess_count / max(1, len(essential))) if essential else 0.0
        des_cov = (des_count / max(1, len(desirable))) if desirable else 0.0

        # ---------- Experience Boosting ----------
        freq_map = {}
        years_map = {}
        # count frequencies for all matched skills
        for sk in ess_matched + des_matched:
            freq_map[sk] = count_skill_frequency(rtext_clean, sk)
            years_map[sk] = detect_years_of_experience(resumes_raw.get(name, ""), sk)

        # Frequency boost: 0.01 per mention across all matched skills
        frequency_boost = 0.01 * sum(freq_map.values())

        # Years boost: 0.02 per year per skill (cap to avoid explosion, e.g., 10 years cap)
        years_boost = 0.0
        for sk, yrs in years_map.items():
            if yrs > 0:
                years_boost += 0.02 * min(yrs, 10)

        # ---------- Section Weighting ----------
        sections = split_sections(resumes_raw.get(name, ""))
        section_score = 0.0
        for sec, txt in sections.items():
            hits = [sk for sk in (essential + desirable) if re.search(rf"(?i)\b{re.escape(sk)}\b", txt or "")]
            if hits:
                section_score += SECTION_WEIGHTS.get(sec, 1.0) * 0.02 * len(hits)

        # Final score (keeps your normalized primary weights + adds bonuses)
        score = norm[0] * sim + norm[1] * ess_cov + norm[2] * des_cov + frequency_boost + years_boost + section_score

        row = {
            "Candidate": name,
            "JD_Match_%": to_percent(sim),
            "Essential_Coverage_%": to_percent(ess_cov),
            "Desirable_Coverage_%": to_percent(des_cov),
            "Total_Score_%": to_percent(score),
            "Matched_Essential": ", ".join(ess_matched) if ess_matched else "",
            "Missing_Essential": ", ".join(ess_missing) if ess_missing else "",
            "Matched_Desirable": ", ".join(des_matched) if des_matched else "",
            "Missing_Desirable": ", ".join(des_missing) if des_missing else "",
        }
        rows.append(row)

        explanation = build_explanation(
            name, row, resumes_raw.get(name, ""), essential, desirable, freq_map, years_map, sections
        )
        explanations.append(explanation)

    df = pd.DataFrame(rows).sort_values(by="Total_Score_%", ascending=False).reset_index(drop=True)
    return df, explanations

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
            "resumes": {}, "results": [], "explanations": [],
            "weights": {"w_jd": 0.5, "w_ess": 0.4, "w_des": 0.1}
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
- Final score = weighted sum of the three (normalized) **plus** Experience & Section bonuses.
- You can also view JD/resume text with **highlights** for matched skills.
- Scanned PDFs are handled using **OCR**.
- Results (and explanations) are saved with the project for later review.
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
        explanations = []

        # normalized weights (same as caption)
        total_w = w_jd + w_ess + w_des
        if total_w == 0:
            total_w = 1.0
        norm = (w_jd / total_w, w_ess / total_w, w_des / total_w)

        for name, rtext_clean in resumes_clean.items():
            sim = compute_similarity(jd_clean, rtext_clean, vectorizer)

            ess_count, ess_matched, ess_missing = skill_matches(rtext_clean, essential)
            des_count, des_matched, des_missing = skill_matches(rtext_clean, desirable)

            ess_cov = (ess_count / max(1, len(essential))) if essential else 0.0
            des_cov = (des_count / max(1, len(desirable))) if desirable else 0.0

            # ---------- Experience Boosting ----------
            freq_map = {}
            years_map = {}
            for sk in ess_matched + des_matched:
                freq_map[sk] = count_skill_frequency(rtext_clean, sk)
                years_map[sk] = detect_years_of_experience(resumes_raw.get(name, ""), sk)

            frequency_boost = 0.01 * sum(freq_map.values())
            years_boost = 0.0
            for sk, yrs in years_map.items():
                if yrs > 0:
                    years_boost += 0.02 * min(yrs, 10)

            # ---------- Section Weighting ----------
            sections = split_sections(resumes_raw.get(name, ""))
            section_score = 0.0
            for sec, txt in sections.items():
                hits = [sk for sk in (essential + desirable) if re.search(rf"(?i)\b{re.escape(sk)}\b", txt or "")]
                if hits:
                    section_score += SECTION_WEIGHTS.get(sec, 1.0) * 0.02 * len(hits)

            score = norm[0] * sim + norm[1] * ess_cov + norm[2] * des_cov + frequency_boost + years_boost + section_score

            row = {
                "Candidate": name,
                "JD_Match_%": to_percent(sim),
                "Essential_Coverage_%": to_percent(ess_cov),
                "Desirable_Coverage_%": to_percent(des_cov),
                "Total_Score_%": to_percent(score),
                "Matched_Essential": ", ".join(ess_matched) if ess_matched else "",
                "Missing_Essential": ", ".join(ess_missing) if ess_missing else "",
                "Matched_Desirable": ", ".join(des_matched) if des_matched else "",
                "Missing_Desirable": ", ".join(des_missing) if des_missing else "",
            }
            rows.append(row)

            explanation = build_explanation(
                name, row, resumes_raw.get(name, ""), essential, desirable, freq_map, years_map, sections
            )
            explanations.append(explanation)

        df = pd.DataFrame(rows).sort_values(by="Total_Score_%", ascending=False).reset_index(drop=True)

        st.subheader("📊 Ranked Candidates")
        st.dataframe(
            df[["Candidate", "Total_Score_%", "JD_Match_%", "Essential_Coverage_%", "Desirable_Coverage_%"]],
            use_container_width=True
        )

        st.markdown(make_download_link(df), unsafe_allow_html=True)

        # Explanation download (TXT)
        all_explanations_txt = "\n\n".join(explanations)
        st.markdown("### 📑 Candidate Explanations")
        st.text_area("Explanations", all_explanations_txt, height=280)
        st.markdown(make_download_text_link(all_explanations_txt, "candidate_explanations.txt"), unsafe_allow_html=True)

        # Save project state (merged resumes) + weights + explanations
        if "project" in st.session_state:
            save_project(st.session_state["project"], {
                "jd": jd,
                "essential": essential,
                "desirable": desirable,
                "resumes": resumes_raw,   # merged
                "results": df.to_dict(orient="records"),
                "explanations": explanations,
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
                        skills_all = parse_skill_list(essential_raw) + parse_skill_list(desirable_raw)
                        jd_highlighted = highlight_text(jd, skills_all, "lightblue")
                        resume_highlighted = highlight_text(
                            (resumes_raw.get(r['Candidate']) or ""), 
                            skills_all, 
                            "lightgreen"
                        )

                        st.markdown("**Job Description with highlights:**", unsafe_allow_html=True)
                        st.markdown(f"<div style='white-space: pre-wrap;'>{jd_highlighted}</div>", unsafe_allow_html=True)

                        st.markdown("**Resume with highlights:**", unsafe_allow_html=True)
                        st.markdown(f"<div style='white-space: pre-wrap;'>{resume_highlighted}</div>", unsafe_allow_html=True)

                    # Explanation panel
                    try:
                        idx = list(df["Candidate"]).index(r["Candidate"])
                        st.markdown("**Explanation:**")
                        st.text_area(f"Explanation for {r['Candidate']}", explanations[idx], height=180, key=f"ex_{r['Candidate']}")
                    except:
                        pass

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

                # show saved weights summary (normalized for display)
                sw = saved.get("weights", {"w_jd": 0.5, "w_ess": 0.4, "w_des": 0.1})
                total_sw = sw["w_jd"] + sw["w_ess"] + sw["w_des"]
                if total_sw <= 0:
                    total_sw = 1.0
                st.caption(f"Saved Weights → JD: {sw['w_jd']/total_sw:.2f}, Essential: {sw['w_ess']/total_sw:.2f}, Desirable: {sw['w_des']/total_sw:.2f}")

                # Download saved explanations
                saved_expls = saved.get("explanations", [])
                if saved_expls:
                    st.markdown(make_download_text_link("\n\n".join(saved_expls), filename=f"{proj}_explanations.txt"), unsafe_allow_html=True)

                # Candidate details with explanation + delete button
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

                            # Highlights
                            if st.checkbox(f"Show JD/Resume text with highlights for {r['Candidate']}", key=f"saved_hl_{r['Candidate']}"):
                                skills_all = saved.get("essential", []) + saved.get("desirable", [])
                                jd_highlighted = highlight_text(saved.get("jd", ""), skills_all, "lightblue")
                                resume_text = (saved.get("resumes", {}).get(r["Candidate"]) or "")
                                resume_highlighted = highlight_text(resume_text, skills_all, "lightgreen")
                                st.markdown("**Job Description with highlights:**", unsafe_allow_html=True)
                                st.markdown(f"<div style='white-space: pre-wrap;'>{jd_highlighted}</div>", unsafe_allow_html=True)
                                st.markdown("**Resume with highlights:**", unsafe_allow_html=True)
                                st.markdown(f"<div style='white-space: pre-wrap;'>{resume_highlighted}</div>", unsafe_allow_html=True)

                            # Explanation (saved)
                            try:
                                idx = list(df_saved["Candidate"]).index(r["Candidate"])
                                ex_saved = saved_expls[idx] if idx < len(saved_expls) else ""
                                st.markdown("**Explanation:**")
                                st.text_area(f"Explanation for {r['Candidate']}", ex_saved, height=180, key=f"saved_ex_{r['Candidate']}")
                            except:
                                pass

                            # Delete candidate button
                            col_del1, col_del2 = st.columns([1,4])
                            with col_del1:
                                if st.button(f"Delete Candidate", key=f"del_{r['Candidate']}"):
                                    # Remove candidate from resumes, recompute, save, refresh
                                    resumes_raw = dict(saved.get("resumes", {}))
                                    if r["Candidate"] in resumes_raw:
                                        resumes_raw.pop(r["Candidate"])
                                        # Recompute with saved data (using same weights)
                                        df_new, expl_new = recompute_results(
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
                                            "explanations": expl_new,
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
