"""Benchmark verification page: gold page image next to the question and evidence.

No retrieval happens here; it only helps check every evidence/answer against the
original scan before setting meta.status = FROZEN.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import pymupdf
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import DEFAULT_MANIFEST, REPO_ROOT, load_corpus  # noqa: E402

QUESTIONS = REPO_ROOT / "benchmark" / "questions.json"
st.set_page_config(page_title="Kiểm chứng benchmark", page_icon="✅", layout="wide")


@st.cache_data
def manifest() -> dict:
    return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["sources"][0]


@st.cache_data
def page_texts() -> dict[int, str]:
    return {p.page: p.text for p in load_corpus(DEFAULT_MANIFEST)[0]}


@st.cache_data
def page_png(page: int, dpi: int = 110) -> bytes:
    pdf = REPO_ROOT / manifest()["local_path"]
    with pymupdf.open(pdf) as doc:
        return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")


bench = json.loads(QUESTIONS.read_text(encoding="utf-8"))
questions = bench["questions"]
st.title("✅ Kiểm chứng benchmark")
st.caption(f"Trạng thái: **{bench['meta']['status'][:60]}** · chỉ hiển thị nguồn, không truy xuất. "
           "Sửa trực tiếp benchmark/questions.json, rồi chạy scripts/validate_benchmark.py.")

if "checked" not in st.session_state:
    st.session_state.checked = {}
done = sum(st.session_state.checked.get(q["id"], False) for q in questions)
st.progress(done / len(questions), text=f"Đã đánh dấu kiểm tra {done}/{len(questions)} câu "
                                        "(chỉ lưu trong phiên trình duyệt này)")

labels = {f"{q['id']} ({q['split']}) · {q['question'][:70]}": q for q in questions}
q = labels[st.selectbox("Câu hỏi", list(labels))]
pdf_exists = (REPO_ROOT / manifest()["local_path"]).exists()

left, right = st.columns([1, 1])
with left:
    st.markdown(f"**{q['id']}** · `{q['split']}` · loại `{q.get('type')}` · "
                f"filter `{(q.get('filter') or {}).get('section', '-')}`")
    st.markdown(f"**Câu hỏi:** {q['question']}")
    st.markdown(f"**Đáp án kỳ vọng:** {q['expected_answer']}")
    st.markdown(f"**Evidence:** _{q['evidence_quote']}_")
    if q.get("note"):
        st.warning(q["note"])
    st.session_state.checked[q["id"]] = st.checkbox(
        "Đã đối chiếu với bản gốc: evidence đúng, đủ điều kiện/ngoại lệ, đáp án đúng",
        value=st.session_state.checked.get(q["id"], False), key=f"chk_{q['id']}")
    for page in q["gold_pages"]:
        text = page_texts().get(page, "")
        pattern = r"\s+".join(map(re.escape, q["evidence_quote"].split()))
        m = re.search(pattern, text, flags=re.I)
        body = (html.escape(text[:m.start()]) + "<mark>" + html.escape(text[m.start():m.end()])
                + "</mark>" + html.escape(text[m.end():])) if m else html.escape(text)
        st.markdown(f"**Text OCR trang {page}** {'✅ có evidence' if m else '❌ không thấy evidence'}")
        st.markdown("<pre style='white-space:pre-wrap;font-family:inherit;font-size:0.8rem;"
                    f"max-height:520px;overflow:auto'>{body}</pre>", unsafe_allow_html=True)
with right:
    for page in q["gold_pages"]:
        st.markdown(f"**Ảnh gốc trang {page}** · [mở PDF]"
                    f"({manifest()['source_path_or_url']}#page={page})")
        if pdf_exists:
            st.image(page_png(page), width='stretch')
        else:
            st.info("Không có PDF local; dùng link để mở bản gốc.")
