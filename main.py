# app.py — PDF 기반 Q&A (UI 단순화 버전)
import os, re, json, tempfile, hashlib, pathlib
from typing import List, Tuple
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import pandas as pd
import altair as alt

# LangChain / OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_community.document_loaders import PyPDFLoader
except ModuleNotFoundError:
    from langchain_community.document_loaders.pdf import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# =========================
# 기본 UI 및 페이지 설정
# =========================
st.set_page_config(page_title="PDF Q&A (Simple UI)", page_icon="📄", layout="centered")
st.title("📄 PDF 기반 AI Q&A")

# =========================
# >>> NEW: 내부 표준 설정값 (사이드바 제거) <<<
# =========================
# 인덱싱 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# 검색/생성 설정
TOP_K = 6
FETCH_K = 20
USE_COMPRESSION = True
TEMPERATURE = 0.1
MAX_TOKENS = 1000
# 비판/리스크 질문 처리(전역 스캔) 설정
ENABLE_GLOBAL_CRITIQUE = True
GLOBAL_CRITIQUE_PAGES_CHARS = 700
GLOBAL_CRITIQUE_TOTAL_CHARS = 12000

# =========================
# API 키
# =========================
def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if key: return key
    try: return st.secrets["OPENAI_API_KEY"]
    except Exception: return None

OPENAI_API_KEY = get_openai_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 없습니다. (.env 또는 .streamlit/secrets.toml / 클라우드 Secrets 설정)")
    st.stop()

# ===== 모델 설정 (환경변수로 오버라이드 가능) =====
PRIMARY_MODEL = os.getenv("OPENAI_CHAT_MODEL_PRIMARY", "gpt-4o-mini")
LIGHT_MODEL   = os.getenv("OPENAI_CHAT_MODEL_LIGHT",   "gpt-4o-mini")
EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL",        "text-embedding-3-small")

# ... (이전 코드의 유틸리티, 인덱싱, LLM 헬퍼, 답변 생성기 함수들은 그대로 유지) ...
# =========================
# 유틸 (의도·개수 파싱 / 비용·캐시)
# =========================
PROS_KEYS = ["잘된", "잘 된", "장점", "강점", "좋은 점"]
CONS_KEYS = ["부족", "단점", "한계", "리스크", "문제점", "취약", "불편", "제약", "위험", "보완"]
def extract_count(q: str, default=3) -> int:
    m = re.search(r"(\d+)\s*가지", q or "")
    return int(m.group(1)) if m else default
def detect_intent(q: str):
    ql = (q or "").lower()
    has_pros = any(k in ql for k in [*PROS_KEYS, "pros", "advantages"])
    has_cons = any(k in ql for k in [*CONS_KEYS, "cons", "risks"])
    if has_pros and has_cons: return "pros_cons"
    if has_cons: return "critique"
    return "general"
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]
def save_dir_for(hash_key: str) -> str:
    base = pathlib.Path(".faiss_cache")
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"vs_{hash_key}")
def extract_page_citations(text: str) -> List[int]:
    pages = set(int(p) for p in re.findall(r"\[p\.(\d+)\]", text or ""))
    return sorted(list(pages))
def to_text(resp) -> str:
    if resp is None: return ""
    try:
        text = (getattr(resp, "content", None) or "").strip()
        if text: return text
        ak = getattr(resp, "additional_kwargs", {}) or {}
        tcs = ak.get("tool_calls") or []
        if tcs:
            fn = (tcs[0] or {}).get("function", {}); args = fn.get("arguments", "") or ""
            return str(args).strip()
        fc = ak.get("function_call") or {}
        if fc: return str(fc.get("arguments", "") or "").strip()
        return ""
    except Exception: return ""
# =========================
# 인덱스 생성/로딩 (캐시 + 디스크 저장)
# =========================
@st.cache_resource(show_spinner=False)
def load_or_build_index(pdf_bytes: bytes, chunk_size:int, chunk_overlap:int, api_key:str):
    pdf_hash = sha256_bytes(pdf_bytes)
    folder = save_dir_for(pdf_hash)
    if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "index.faiss")):
        embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)
        vs = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes); path = tmp.name
        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])
        splits = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)
        vs = FAISS.from_documents(splits, embeddings)
        vs.save_local(folder)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes); path2 = tmp.name
    page_docs = PyPDFLoader(path2).load()
    pages = [(d.metadata.get("page", 0)+1, d.page_content) for d in page_docs]
    return vs, pages
def _format_docs(docs, max_chars=4000):
    return "\n\n".join(f"[p.{(d.metadata.get('page',0)+1)}] {d.page_content}" for d in docs)[:max_chars]
# =========================
# LLM 호출 헬퍼
# =========================
def llm_chat(max_tokens:int=600, temperature:float=0.2):
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=PRIMARY_MODEL, temperature=temperature, max_tokens=max_tokens)
def llm_light(max_tokens:int=300, temperature:float=0):
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=LIGHT_MODEL, temperature=temperature, max_tokens=max_tokens)
def llm_chat_json(model=PRIMARY_MODEL, max_tokens=700, temperature=0.1):
    return ChatOpenAI(
        api_key=OPENAI_API_KEY, model=model, temperature=temperature, max_tokens=max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
# =========================
# AI 질문 추천 기능
# =========================
@st.cache_data(show_spinner="문서 분석 및 질문 추천 중...")
def generate_question_suggestions(_pages: List[Tuple[int,str]]) -> List[str]:
    context = "\n".join([f"[p.{p}] {txt[:500]}" for p, txt in _pages[:5]])[:4000]
    sys = (
        "You are an AI assistant that helps users understand complex documents. "
        "Based on the provided context from a document, generate three insightful and distinct questions a user might ask. "
        'Return the result as a valid JSON list of strings. Example: ["Question 1?", "Question 2?", "Question 3?"]'
    )
    user = f"CONTEXT:\n{context}"
    try:
        chain = llm_chat_json(model=LIGHT_MODEL, max_tokens=300) | JsonOutputParser()
        questions = chain.invoke([{"role":"system","content":sys}, {"role":"user","content":user}])
        if isinstance(questions, list) and len(questions) > 0:
            return questions[:3]
    except Exception: pass
    return ["이 문서의 핵심 내용을 3가지로 요약해줘", "주요 리스크나 우려되는 점은 무엇이야?", "이 문서가 제시하는 향후 계획은 뭐야?"]
# =========================
# AI 시각화 기능
# =========================
def generate_visualization_code(question: str, context: str) -> dict:
    sys = """
You are a data visualization expert. Your task is to generate Altair chart code based on a user's question and a provided text context.
1. First, determine if the question can be answered with a chart using the given context. Look for numerical data, trends, comparisons, or proportions.
2. If visualization is not possible or the data is insufficient, return a JSON object with a single key: `{"error": "No suitable data for visualization."}`.
3. If visualization is possible:
    a. Extract the necessary data and structure it as a JSON list of objects.
    b. Write Python code using the Altair library to create a chart. The code must use a pandas DataFrame named `df` which will be created from the data.
    c. Choose the best chart type (bar, line, pie, area, etc.) to answer the question. Make sure chart labels and titles are in KOREAN.
4. Return a single valid JSON object with two keys: "data" (the extracted data) and "code" (the Altair code string).
"""
    user = f"USER_QUESTION: \"{question}\"\n\nCONTEXT:\n{context}"
    try:
        chain = llm_chat_json(max_tokens=800) | JsonOutputParser()
        result = chain.invoke([{"role":"system","content":sys}, {"role":"user","content":user}])
        return result
    except Exception:
        return {"error": "Failed to generate visualization code."}

def safe_execute_altair_code(code_str: str, data: list):
    if not data or not isinstance(data, list): return None
    try:
        df = pd.DataFrame(data)
        local_scope = {"alt": alt, "pd": pd, "df": df}
        exec(code_str, {}, local_scope)
        chart = local_scope.get("chart")
        if chart and isinstance(chart, alt.TopLevelMixin):
            return chart
        return None
    except Exception: return None
# =========================
# 상태 초기화
# =========================
if "vs" not in st.session_state: st.session_state.vs = None
if "pages" not in st.session_state: st.session_state.pages = None
if "messages" not in st.session_state: st.session_state.messages = []
if "uploaded_name" not in st.session_state: st.session_state.uploaded_name = None
if "suggested_questions" not in st.session_state: st.session_state.suggested_questions = []

# =========================
# 메인 UI
# =========================
uploaded = st.file_uploader("1. 분석할 PDF 문서를 업로드하세요.", type=["pdf"])
use_visualization = st.checkbox("2. 답변 시각화 기능 사용 (베타)", value=False, help="답변에 표나 수치가 있을 경우 AI가 차트를 함께 생성합니다.")

if uploaded:
    if st.session_state.uploaded_name != uploaded.name:
        with st.spinner("PDF 분석 및 인덱싱 중..."):
            vs, pages = load_or_build_index(uploaded.read(), CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY)
            st.session_state.vs, st.session_state.pages = vs, pages
            st.session_state.uploaded_name = uploaded.name
            st.session_state.messages = []
            st.session_state.suggested_questions = generate_question_suggestions(pages)
        st.success(f"'{uploaded.name}' 분석 완료! 아래 추천 질문을 클릭하거나 직접 질문을 입력하세요.")
else:
    st.session_state.vs, st.session_state.pages, st.session_state.uploaded_name, st.session_state.messages, st.session_state.suggested_questions = None, None, None, [], []

# =========================
# 대화형 체인 빌더
# =========================
def build_chain(retriever):
    system = "너는 업로드된 PDF에 근거해 한국어로 답한다. 이전 대화 내용을 참고하여, 현재 질문에 가장 적절한 답변을 생성한다. 항상 **사용자 질문의 의도와 개수 요구를 정확히 따른다**. 가능하면 bullet을 사용하고, 각 핵심 주장 끝에 [p.페이지] 근거를 붙인다. 문서와 무관한 추측은 금지한다."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system + "\n\nCONTEXT:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    chain = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever | _format_docs)
        | {"answer": prompt | llm_chat(max_tokens=MAX_TOKENS, temperature=TEMPERATURE) | StrOutputParser(), "context": itemgetter("context")}
    )
    return chain
# =========================
# 질문 처리 로직 함수
# =========================
def handle_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            base_retriever = st.session_state.vs.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": FETCH_K})
            retriever = base_retriever
            if USE_COMPRESSION:
                compressor = LLMChainExtractor.from_llm(llm_light(max_tokens=300, temperature=0))
                retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)

            chain = build_chain(retriever)
            chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages[:-1]]
            result = chain.invoke({"question": question, "chat_history": chat_history})
            answer = result.get("answer", "답변을 생성하지 못했습니다.")
            context = result.get("context", "")

            if not (answer or "").strip(): answer = "⚠️ 응답이 비어 있습니다."
            st.write(answer)

            if use_visualization:
                with st.spinner("시각화 생성 중..."):
                    viz_result = generate_visualization_code(question, context)
                    if "error" not in viz_result:
                        chart = safe_execute_altair_code(viz_result.get("code"), viz_result.get("data"))
                        if chart: st.altair_chart(chart, use_container_width=True)
                        else: st.info("차트를 생성하는 데 실패했습니다.")
                    else: st.info(f"시각화 정보: {viz_result['error']}")

            cited_pages = extract_page_citations(answer)
            if cited_pages and st.session_state.pages:
                with st.expander(f"🔎 인용된 페이지 스니펫 보기 ({len(cited_pages)}개)"):
                    page_map = {pg: txt for pg, txt in st.session_state.pages}
                    for pg in cited_pages: st.markdown(f"**[p.{pg}]**\n\n{(page_map.get(pg) or '')[:500]}")
    st.session_state.messages.append({"role": "assistant", "content": answer})

# =========================
# Q&A 및 대화 기록 UI
# =========================
if st.session_state.vs:
    st.markdown("---")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.suggested_questions and len(st.session_state.messages) == 0:
        st.markdown("##### AI 추천 질문:")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, question_prompt in enumerate(st.session_state.suggested_questions):
            with cols[i]:
                if st.button(question_prompt, key=f"suggestion_{i}", use_container_width=True):
                    handle_question(question_prompt)
                    st.rerun()
    
    if user_input := st.chat_input("PDF 내용에 대해 질문하세요..."):
        handle_question(user_input)
        st.rerun()

else:
    st.info("PDF 문서를 업로드하면 AI 분석이 시작됩니다.")