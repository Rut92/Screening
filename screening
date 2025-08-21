# resume_screening_app.py
# Run: pip install streamlit scikit-learn PyPDF2 python-docx pandas numpy pytesseract pdf2image pillow
# Then: streamlit run resume_screening_app.py

import io
import re
import os
import math
import base64
import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OCR
import pytesseract
from pdf2image import convert_from_bytes

# ------------------------------- Helpers -------------------------------- #

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

# ------------------------------- UI ------------------------------------- #

st.set_page_config(page_title="📄 Resume Screening App", layout="wide")
st.title("📄 Resume Screening & Ranking")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
- Enter the **Job Description**, and list **Essential** and **Desirable** skills.
- Upload resumes in **PDF, DOCX, or TXT**.
- The app computes JD match (cosine similarity), essential coverage, and desirable coverage.
- Final score = weighted sum of the three. Adjust weights in the sidebar.
- You can also view JD/resume text with **highlights** for matched skills.
- Scanned PDFs are handled using **OCR**.
        """
    )

colA, colB = st.columns([2, 1])

with colA:
    jd = st.text_area("Job Description", height=220)

    c1, c2 = st.columns(2)
    with c1:
        essential_raw = st.text_area("Essential Skills", height=160)
    with c2:
        desirable_raw = st.text_area("Desirable Skills", height=160)

    files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

with colB:
    st.subheader("Scoring Weights")
    w_jd = st.slider("JD Match Weight", 0.0, 1.0, 0.5, 0.05)
    w_ess = st.slider("Essential Coverage Weight", 0.0, 1.0, 0.4, 0.05)
    w_des = st.slider("Desirable Coverage Weight", 0.0, 1.0, 0.1, 0.05)

    total_w = w_jd + w_ess + w_des
    if total_w == 0:
        norm = (0.0, 0.0, 0.0)
    else:
        norm = (w_jd / total_w, w_ess / total_w, w_des / total_w)

    st.caption(f"Normalized Weights → JD: {norm[0]:.2f}, Essential: {norm[1]:.2f}, Desirable: {norm[2]:.2f}")

    show_details = st.checkbox("Show per-candidate skill details", value=True)

st.markdown("---")

run = st.button("🔎 Screen & Rank Candidates", type="primary", disabled=not (jd and files))

# ------------------------------- Processing ----------------------------- #

if run:
    essential = parse_skill_list(essential_raw)
    desirable = parse_skill_list(desirable_raw)

    jd_clean = clean_text(jd)
    resumes_raw: Dict[str, str] = {}
    resumes_clean: Dict[str, str] = {}

    for f in files:
        text = extract_text_from_file(f)
        resumes_raw[f.name] = text or ""
        resumes_clean[f.name] = clean_text(text or "")

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
                    jd_highlighted = highlight_text(jd, essential + desirable, "lightblue")
                    resume_highlighted = highlight_text(resumes_raw[r['Candidate']], essential + desirable, "lightgreen")

                    st.markdown("**Job Description with highlights:**", unsafe_allow_html=True)
                    st.markdown(f"<div style='white-space: pre-wrap;'>{jd_highlighted}</div>", unsafe_allow_html=True)

                    st.markdown("**Resume with highlights:**", unsafe_allow_html=True)
                    st.markdown(f"<div style='white-space: pre-wrap;'>{resume_highlighted}</div>", unsafe_allow_html=True)
