"""Local chat UI to query the pgvector index and inspect retrieved evidence.

Run from starter/:  .venv\\Scripts\\streamlit run app.py
Retrieval uses exactly the same embedder/store/strategies as src.run.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import OpenAIEmbedder
from src.ingestion import DEFAULT_MANIFEST, REPO_ROOT
from src.run import STRATEGIES
from src.store import PgVectorStore

STARTER = Path(__file__).resolve().parent
load_dotenv(STARTER / ".env")
QUESTIONS = REPO_ROOT / "benchmark" / "questions.json"
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
STRATEGY_LABEL = {
    "A": "A · fixed 500/50",
    "B": "B · paragraph ≤1000",
    "C": "C · B + section filter",
}

st.set_page_config(page_title="MB BCTC Q3/2026 · RAG checker", page_icon="🔎", layout="wide")


# ---------- resources ----------
@st.cache_resource
def get_backend() -> tuple[OpenAIEmbedder, PgVectorStore]:
    embedder = OpenAIEmbedder()
    store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions)
    return embedder, store


@st.cache_data
def load_manifest() -> dict:
    return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))


@st.cache_data
def load_benchmark() -> dict | None:
    if not QUESTIONS.exists():
        return None
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def canon(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def is_hit(hit: dict, gold: dict | None) -> bool | None:
    if not gold:
        return None
    return (hit["document_id"] == gold["gold_source_id"] and hit["page"] in gold["gold_pages"]
            and canon(gold["evidence_quote"]) in canon(hit["text"]))


def run_query(question: str, strategies: list[str], top_k: int, section: str | None) -> dict:
    embedder, store = get_backend()
    t0 = time.perf_counter()
    vector = embedder.embed_query(question)
    embed_ms = (time.perf_counter() - t0) * 1000
    results = {}
    for name in strategies:
        cfg = STRATEGIES[name]
        filters = {"section": section} if cfg["use_filter"] and section else None
        t1 = time.perf_counter()
        hits = store.search(cfg["index"], vector, top_k=top_k, filters=filters)
        results[name] = {"hits": hits, "filters": filters,
                         "search_ms": round((time.perf_counter() - t1) * 1000, 1)}
    return {"embed_ms": round(embed_ms, 1), "results": results}


def generate_answer(question: str, hits: list[dict], model: str) -> str:
    embedder, _ = get_backend()
    context = "\n\n".join(f"[Đoạn {i + 1} | trang {h['page']} | {h['section']}]\n{h['text']}"
                          for i, h in enumerate(hits))
    response = embedder.client.chat.completions.create(
        model=model, temperature=0,
        messages=[
            {"role": "system", "content": (
                "Bạn trả lời câu hỏi về Báo cáo tài chính hợp nhất Quý III/2026 của MB. "
                "Chỉ dùng thông tin trong các đoạn được cung cấp; text là kết quả OCR nên có thể "
                "có lỗi chính tả. Ghi nguồn dạng (trang N) sau mỗi ý. Nếu các đoạn không đủ để trả "
                "lời, nói rõ là không tìm thấy trong ngữ cảnh truy xuất, không tự suy đoán. "
                "Bảng số liệu được dựng lại từ OCR: mỗi dòng là một hàng, các cột ngăn bởi ' | '; "
                "tiêu đề cột (kỳ báo cáo, đơn vị) nằm ở các dòng phía trên. Khi nêu một con số, "
                "nói rõ nó thuộc chỉ tiêu và cột/kỳ nào; nếu không xác định chắc chắn được cột "
                "(ví dụ thiếu dòng tiêu đề trong đoạn) thì nói rõ điều đó. Đơn vị mặc định là "
                "triệu đồng nếu bảng ghi như vậy.")},
            {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"},
        ])
    return response.choices[0].message.content


# ---------- sidebar ----------
manifest = load_manifest()
source = manifest["sources"][0]
section_titles = {s["id"]: f"{s['title']} (tr.{s['pages'][0]}–{s['pages'][1]})"
                  for s in source["sections"]}
bench = load_benchmark()

with st.sidebar:
    st.header("Cấu hình truy xuất")
    mode = st.radio("Chế độ", ["So sánh A · B · C", "Một chiến lược"], horizontal=False)
    if mode == "Một chiến lược":
        strategies = [st.radio("Chiến lược", list(STRATEGIES), format_func=STRATEGY_LABEL.get,
                               horizontal=True)]
    else:
        strategies = list(STRATEGIES)
    top_k = st.slider("Top-k", 1, 10, 3)
    section = st.selectbox("Filter `section` (chỉ áp dụng cho C)", [None, *section_titles],
                           format_func=lambda s: "— không lọc —" if s is None else section_titles[s],
                           key="section")
    if section and "C" not in strategies:
        st.caption("⚠️ Filter chỉ có tác dụng với chiến lược C.")

    st.divider()
    want_answer = st.toggle("Sinh câu trả lời từ top-k (OpenAI)", value=False,
                            help="Chỉ để demo. Benchmark của lab chấm evidence truy xuất, "
                                 "không chấm câu trả lời LLM.")
    chat_model = st.text_input("Chat model", DEFAULT_CHAT_MODEL, disabled=not want_answer)

    st.divider()
    st.subheader("Câu hỏi benchmark")
    gold_pick = None
    if bench:
        frozen = str(bench["meta"].get("status", "")).upper().startswith("FROZEN")
        visible = [q for q in bench["questions"] if q["split"] == "dev" or frozen]
        if not frozen:
            st.caption("🔒 Benchmark chưa FROZEN: chỉ hiện câu **dev**. "
                       "Đừng gõ tay câu test trước khi chốt.")
        options = {f"{q['id']} · {q['question'][:60]}": q for q in visible}
        label = st.selectbox("Chọn câu", [None, *options], format_func=lambda x: x or "—")
        if label and st.button("Hỏi câu này", width='stretch'):
            gold_pick = options[label]
    else:
        st.caption("Chưa có benchmark/questions.json")

    st.divider()
    if st.button("Xóa hội thoại", width='stretch'):
        st.session_state.messages = []
        st.rerun()


# ---------- rendering ----------
def render_chunk_text(text: str, evidence: str | None) -> str:
    """Show OCR text verbatim (no Markdown parsing of '1.', '-', '|', '*' ...).

    The chunk goes into an HTML <pre> block with escaped content; when the chunk is a
    hit, the evidence span is wrapped in <mark> (whitespace-insensitive match).
    """
    span = None
    if evidence:
        pattern = r"\s+".join(map(re.escape, evidence.split()))
        match = re.search(pattern, text, flags=re.I)
        span = match.span() if match else None
    if span:
        body = (html.escape(text[:span[0]]) + "<mark>" + html.escape(text[span[0]:span[1]])
                + "</mark>" + html.escape(text[span[1]:]))
    else:
        body = html.escape(text)
    return ("<pre style='white-space:pre-wrap;font-family:inherit;font-size:0.85rem;"
            f"max-height:420px;overflow:auto;margin:0'>{body}</pre>")


def render_hits(name: str, run: dict, gold: dict | None, compact: bool) -> None:
    hits = run["hits"]
    flags = [is_hit(h, gold) for h in hits]
    head = f"**{STRATEGY_LABEL[name]}**"
    if run["filters"]:
        head += f"  \nfilter: `section={run['filters']['section']}`"
    elif name == "C":
        head += "  \nfilter: _không có (chạy như B)_"
    st.markdown(head)
    if gold:
        hit1 = bool(flags) and bool(flags[0])
        hit3 = any(flags[:3])
        st.markdown(f"Hit@1: {'✅' if hit1 else '❌'} · Evidence Hit@3: {'✅' if hit3 else '❌'}")
    st.caption(f"search {run['search_ms']} ms")
    for rank, (h, flag) in enumerate(zip(hits, flags), start=1):
        mark = "" if flag is None else (" ✅" if flag else "")
        heading = h["metadata"].get("heading")
        title = f"#{rank} · tr.{h['page']} · score {h['score']:.3f}{mark}"
        with st.expander(title, expanded=(rank == 1 and not compact)):
            st.caption(f"`{h['chunk_id']}`  \nsection: `{h['section']}`"
                       + (f"  \nheading: {heading}" if heading else "")
                       + f"  \nOCR conf: {h['metadata'].get('ocr_mean_confidence')}")
            st.markdown(render_chunk_text(h["text"], gold["evidence_quote"] if gold and flag else None),
                        unsafe_allow_html=True)
            st.markdown(f"[Mở trang {h['page']} trong PDF gốc]"
                        f"({source['source_path_or_url']}#page={h['page']})")


def render_assistant(msg: dict) -> None:
    gold = msg.get("gold")
    if gold:
        st.info(f"**Gold** {gold['id']} ({gold['split']}) · trang {gold['gold_pages']}  \n"
                f"Đáp án: {gold['expected_answer']}  \n"
                f"Evidence: _{gold['evidence_quote']}_")
    if msg.get("answer"):
        st.markdown(msg["answer"])
        st.caption(f"Sinh bởi {msg['chat_model']} từ top-k của "
                   f"{STRATEGY_LABEL[msg['answer_strategy']]}")
    runs = msg["results"]
    cols = st.columns(len(runs))
    for col, (name, run) in zip(cols, runs.items()):
        with col:
            render_hits(name, run, gold, compact=len(runs) > 1)
    st.caption(f"embed query {msg['embed_ms']} ms")


# ---------- page ----------
st.title("🔎 MB · BCTC hợp nhất Q3/2026")
st.caption(f"Corpus: {source['counted_content_pages']} trang · embedding "
           f"`{os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}` · pgvector cosine (exact)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

typed = st.chat_input("Hỏi về báo cáo tài chính…")
question = typed or (gold_pick["question"] if gold_pick else None)

if question:
    gold = gold_pick if (gold_pick and not typed) else None
    active_section = section
    if gold is not None:  # benchmark question: use its own pre-declared filter
        active_section = (gold.get("filter") or {}).get("section")
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang truy xuất…"):
                out = run_query(question, strategies, top_k, active_section)
            msg = {"role": "assistant", **out, "gold": gold}
            if want_answer:
                answer_strategy = "C" if "C" in out["results"] else strategies[0]
                with st.spinner("Đang sinh câu trả lời…"):
                    msg["answer"] = generate_answer(
                        question, out["results"][answer_strategy]["hits"], chat_model)
                msg["chat_model"] = chat_model
                msg["answer_strategy"] = answer_strategy
            render_assistant(msg)
            st.session_state.messages.append(msg)
        except Exception as exc:  # show backend errors instead of failing silently
            st.error(f"Lỗi: {type(exc).__name__}: {exc}")
