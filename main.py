# app.py  — PDF 기반 Q&A (강화 RAG, 비용 최적화 + 견고 출력)
import os, re, json, tempfile, hashlib, pathlib
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import pandas as pd

# LangChain / OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_community.document_loaders import PyPDFLoader
except ModuleNotFoundError:
    from langchain_community.document_loaders.pdf import PyPDFLoader  # 아주 구버전 대비

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# (선택) 컨텍스트 압축
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# =========================
# 기본 UI
# =========================
st.set_page_config(page_title="PDF QA (강화 RAG)", page_icon="📄", layout="centered")
st.title("📄 PDF 기반 Q&A (요약표 → 질문/답변, 강화 RAG)")

# =========================
# API 키
# =========================
def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None

OPENAI_API_KEY = get_openai_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 없습니다. (.env 또는 .streamlit/secrets.toml / 클라우드 Secrets 설정)")
    st.stop()

# ===== 모델 설정 (환경변수로 오버라이드 가능) =====
PRIMARY_MODEL = os.getenv("OPENAI_CHAT_MODEL_PRIMARY", "gpt-5-mini")   # 메인 생성/추론
LIGHT_MODEL   = os.getenv("OPENAI_CHAT_MODEL_LIGHT",   "gpt-5-nano")   # 압축/정제/간단 태스크
EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL",         "text-embedding-3-small")  # 최저가 임베딩

# =========================
# 비용 최적화 기본값 (슬라이더)
# =========================
with st.sidebar:
    st.subheader("🔧 인덱싱")
    chunk_size = st.slider("Chunk size", 300, 2000, 900, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
    st.caption("너무 작으면 청크 수↑(임베딩 비용↑), 너무 크면 검색 정밀도↓")

    st.subheader("🔍 검색/생성")
    top_k = st.slider("검색 결과 개수 k", 1, 10, 5, 1)
    fetch_k = st.slider("후보 fetch_k", 10, 60, 20, 2)
    use_compression = st.checkbox("컨텍스트 압축 사용(권장, 비용↓·정확도↑)", True)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("응답 토큰 상한", 200, 1200, 450, 50)

    st.subheader("🧠 비판/리스크 질문 처리(전역 스캔)")
    enable_global_critique = st.checkbox("비판형 질문에 전역 스캔 사용", True)
    global_critique_pages = st.slider("전역 스캔: 페이지당 발췌 길이", 200, 1200, 700, 50)
    global_critique_total = st.slider("전역 스캔: 전체 컨텍스트 길이", 4000, 20000, 12000, 500)

    st.subheader("🧭 컨설팅 모드 (팔란티어 스타일)")
    strategy_mode = st.checkbox("컨설팅 모드 활성화", False)
    strat_goal = st.text_input("핵심 목표(KPI/매출지표)", "월 구독 매출 MRR 증대")
    strat_horizon = st.selectbox("시간축", ["90일", "180일", "12개월"], index=0)
    strat_budget = st.text_input("예산/제약(선택)", "제한적 인력 2명, 마케팅 월 1천만 원")
    strat_segment = st.text_input("핵심 세그먼트(선택)", "초보 식집사, 프리미엄 구독자")
    strat_competitors = st.text_input("경쟁사/대체재(선택)", "플랜트ID, 네이버 식물 커뮤니티")

uploaded = st.file_uploader("PDF 업로드", type=["pdf"])
question = st.text_input("질문을 입력하세요 (예: '잘된 점 3가지와 부족한 점 3가지를 알려줘')")

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
    if has_pros and has_cons: return "pros_cons"  # 장/단점 혼합
    if has_cons: return "critique"               # 단점/리스크 중심
    return "general"

def looks_like_critique(q: str) -> bool:
    return detect_intent(q) == "critique"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def save_dir_for(hash_key: str) -> str:
    base = pathlib.Path(".faiss_cache")
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"vs_{hash_key}")

def extract_page_citations(text: str) -> List[int]:
    pages = set(int(p) for p in re.findall(r"\[p\.(\d+)\]", text or ""))
    return sorted(list(pages))

# ---- 견고한 JSON 파서 ----
def safe_json_loads(s: str, allow_empty: bool = False):
    if not s or not s.strip():
        if allow_empty:
            return {}
        raise ValueError("empty JSON response")
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL)
    m = re.search(r"\{[\s\S]*\}", s)
    if m: s = m.group(0)
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    s = re.sub(r"(?m)^\s*//.*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        if allow_empty:
            return {}
        raise

# ---- LLM 응답에서 텍스트 강제 추출 (tool_calls 대비) ----
def to_text(resp) -> str:
    """
    BaseMessage에서 content가 비었을 때 tool_calls/function_call.arguments에서 본문 회수.
    항상 문자열 반환.
    """
    if resp is None:
        return ""
    try:
        text = (getattr(resp, "content", None) or "").strip()
        if text:
            return text
        ak = getattr(resp, "additional_kwargs", {}) or {}
        # OpenAI tool_calls
        tcs = ak.get("tool_calls") or []
        if tcs:
            fn = (tcs[0] or {}).get("function", {})
            args = fn.get("arguments", "") or ""
            return str(args).strip()
        # legacy function_call
        fc = ak.get("function_call") or {}
        if fc:
            return str(fc.get("arguments", "") or "").strip()
        return ""
    except Exception:
        return ""

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
            tmp.write(pdf_bytes)
            path = tmp.name
        docs = PyPDFLoader(path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
        )
        splits = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)
        vs = FAISS.from_documents(splits, embeddings)
        vs.save_local(folder)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path2 = tmp.name
    page_docs = PyPDFLoader(path2).load()
    pages = [(d.metadata.get("page", 0)+1, d.page_content) for d in page_docs]
    return vs, pages

def summarize_corpus(retriever, pages, max_chars=10000):
    """PDF 기반 핵심 사실 요약(시장/제품/수익/강약점/리스크) + 페이지 근거 포함"""
    # 전역 + 국소 섞기 (비용 절약을 위해 상한 적용)
    context = []
    if pages:
        for p, txt in pages[:10]:  # 앞쪽 10페이지만 슬림하게
            context.append(f"[p.{p}] {txt[:600]}")
    docs = retriever.get_relevant_documents("서비스 개요, 시장, 수익모델, 강점/약점, 리스크, 실행 계획")
    context.extend([f"[p.{d.metadata.get('page',0)+1}] {d.page_content}" for d in docs])
    context = "\n\n".join(context)[:max_chars]

    sys = (
        "너는 자료조사 분석가다. CONTEXT에서 다음을 항목별로 요약하라."
        "\n1) 시장/고객\n2) 제품/핵심가치\n3) 수익모델\n4) 강점\n5) 약점/리스크\n6) 현재 가정·제약"
        "\n각 bullet 끝에 [p.x] 근거를 붙여라. 한국어로 간결하게."
    )
    user = "CONTEXT:\n" + context
    return to_text(llm_light(max_tokens=650, temperature=0).invoke(
        [{"role":"system","content":sys},{"role":"user","content":user}]
    ))

def analyze_strategy(brief:str, goal:str, horizon:str, budget:str, segment:str, competitors:str):
    """컨설팅 모드: 수치/로드맵/KPI/리스크까지 구조화된 전략 리포트(JSON) 생성"""
    sys = (
        "너는 데이터 중심 전략 컨설턴트다. 아래 BRIEF(자료요약)를 바탕으로 "
        "매출 극대화 전략을 **수치 가설과 실행 로드맵**으로 제시하라. "
        "각 제안은 명확한 가정/근거와 리스크/대응을 포함해야 한다. "
        "출력은 JSON 한 개만 반환하라."
    )
    schema = (
        '{'
        '"summary":"고수준 요약",'
        '"north_star":"핵심 목표/KPI",'
        '"assumptions":["가정 ..."],'
        '"strategies":[{'
            '"name":"전략명",'
            '"rationale":"왜 이 전략인지",'
            '"expected_impact":{"metric":"지표명","baseline":"현재값","target":"목표값","lift_pct":10},'
            '"actions":{"90d":["액션1","액션2"],"180d":["..."],"12m":["..."]},'
            '"kpis":["KPI1","KPI2"],'
            '"risks":[{"risk":"위험","mitigation":"대응"}],'
            '"evidence":["[p.x]","[p.y]"]'
        '}],'
        '"data_needs":["추가로 필요한 데이터/실험"]'
        '}'
    )
    user = (
        f"BRIEF:\n{brief}\n\n"
        f"목표(North Star): {goal}\n"
        f"시간축: {horizon}\n"
        f"예산/제약: {budget}\n"
        f"핵심 세그먼트: {segment}\n"
        f"경쟁사: {competitors}\n\n"
        f"위 조건을 반영해 JSON 스키마 {schema} 로 반환하라. "
        "각 수치 목표는 보수적/현실적 가정으로 추정치를 제시하라(예: 전환율 +x%p, ARPU +y%). "
        "문서 근거는 evidence에 [p.x] 형태로 포함."
    )
    # JSON 모드 강제
    raw = to_text(llm_chat_json(max_tokens=900, temperature=0).invoke(
        [{"role":"system","content":sys},{"role":"user","content":user}]
    ))
    data = safe_json_loads(raw, allow_empty=True)
    if not isinstance(data, dict) or not data:
        # 마지막 폴백: 마크다운
        return {
            "summary": "분석 결과(텍스트 폴백)",
            "strategies": [{"name":"전략 제안 생성 실패", "rationale": raw[:400] or "빈 응답"}],
        }
    return data


def _format_docs(docs, max_chars=3000):
    text = "\n\n".join(f"[p.{(d.metadata.get('page',0)+1)}] {d.page_content}" for d in docs)
    return text[:max_chars]

# =========================
# LLM 호출 헬퍼 (5 mini / 5 nano / JSON 모드)
# =========================
def llm_chat(max_tokens:int=600, temperature:float=0.2):
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=PRIMARY_MODEL,
                      temperature=temperature, max_tokens=max_tokens)

def llm_light(max_tokens:int=300, temperature:float=0):
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=LIGHT_MODEL,
                      temperature=temperature, max_tokens=max_tokens)

def llm_chat_json(max_tokens:int=700, temperature:float=0):
    # json_object 강제 + 실패 시 일반모드로 JSON만 출력하도록 폴백
    def _invoke(msgs):
        try:
            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=PRIMARY_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"response_format": {"type": "json_object"}},
            ).invoke(msgs).content
        except Exception:
            hard_sys = (
                '다음 요청에 대해 {"pros":["..."],"cons":["..."]} 형식의 **유효한 JSON 한 개만** 출력하라. '
                "그 외 텍스트/코드블록/설명 금지."
            )
            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=PRIMARY_MODEL,
                temperature=0,
                max_tokens=max_tokens
            ).invoke(
                [{"role":"system","content":hard_sys}] + msgs
            ).content
    class _Runner:
        def invoke(self, msgs):
            return type("resp", (), {"content": _invoke(msgs)})()
    return _Runner()

# =========================
# pros/cons 전용 루트: JSON → 마크다운 (3중 안전망)
# =========================
def answer_pros_cons(question: str, retriever, pages, n: int, use_global: bool,
                     per_page_chars:int, total_chars:int, api_key:str, max_tokens:int=750) -> str:
    # 컨텍스트 구성
    if use_global and pages:
        buf = [f"[p.{p}] {txt[:per_page_chars]}" for p, txt in pages]
        context = "\n\n".join(buf)[:total_chars]
    else:
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(
            f"[p.{d.metadata.get('page',0)+1}] {d.page_content}" for d in docs
        )[: min(12000, total_chars)]

    sys = (
        "다음 CONTEXT에 근거해 사용자의 질문(장점과 단점을 각각 N개)을 정확히 따른다. "
        "각 항목의 문장 끝에 반드시 [p.x] 근거를 포함한다. "
        '출력은 {"pros":["... [p.x]"],"cons":["... [p.y]"]} 형식의 JSON **한 개**만 반환하라.'
    )
    user = f"N={n}\n질문: {question}\n\nCONTEXT:\n{context}"

    # 1) JSON 모드 1차 + 미세 재시도
    data = {}
    out = ""
    for _ in range(2):
        out = to_text(
            llm_chat_json(max_tokens=max_tokens, temperature=0).invoke(
                [{"role":"system","content":sys},{"role":"user","content":user}]
            )
        )
        data = safe_json_loads(out, allow_empty=True)
        if data:
            break

    # 2) 경량(nano)로 JSON 추출/수정
    if not data:
        out2 = to_text(
            llm_light(max_tokens=300, temperature=0).invoke(
                [
                    {"role":"system","content":"아래 텍스트에서 필요한 JSON만 추출/수정하여 유효한 JSON 한 개로 반환하라."},
                    {"role":"user","content":out or ""},
                ]
            )
        )
        data = safe_json_loads(out2, allow_empty=True)

    # 3) 그래도 실패 → 마크다운 직접 생성
    if not isinstance(data, dict) or ("pros" not in data and "cons" not in data):
        md = to_text(
            llm_chat(max_tokens=max_tokens).invoke(
                [
                    {"role":"system","content":f"장점 {n}개와 단점 {n}개를 각각 bullet로 생성하라. 각 bullet 끝에 [p.x]를 붙여라. 문서 외 추측 금지."},
                    {"role":"user","content":f"질문: {question}\n\nCONTEXT:\n{context}"},
                ]
            )
        )
        return md

    pros = list(data.get("pros", []))[:n]
    cons = list(data.get("cons", []))[:n]
    if not pros: pros = ["문서 기반 장점 요약 [p.?]"]
    if not cons: cons = ["문서 기반 부족/리스크 요약 [p.?]"]

    return "\n".join([
        "#### ✅ 잘된 점", *[f"- {p}" for p in pros], "",
        "#### ⚠️ 부족한 점", *[f"- {c}" for c in cons]
    ])

# =========================
# 전역 스캔(비판형): 질문을 그대로 따름
# =========================
def critique_answer_global(pages: List[Tuple[int,str]], per_page_chars:int, total_chars:int,
                           api_key:str, question:str, max_tokens:int=700) -> str:
    buf = [f"[p.{p}] {txt[:per_page_chars]}" for p, txt in pages]
    context = "\n\n".join(buf)[:total_chars]
    sys = (
        "다음 CONTEXT에 근거해 **사용자 질문을 정확히 따른다**. "
        "가능하면 bullet로 간결히 서술하고, 각 항목 끝에 [p.x]를 붙인다. "
        "문서 외 추측 금지."
    )
    user = f"질문: {question}\n\nCONTEXT:\n{context}"
    return to_text(
        llm_chat(max_tokens=max_tokens).invoke(
            [{"role":"system","content":sys},{"role":"user","content":user}]
        )
    )

# =========================
# 상태
# =========================
if "vs" not in st.session_state: st.session_state.vs = None
if "pages" not in st.session_state: st.session_state.pages = None
if "digest_df" not in st.session_state: st.session_state.digest_df = None

# =========================
# 업로드 처리
# =========================
if uploaded is not None and (st.session_state.get("uploaded_name") != uploaded.name):
    with st.spinner("PDF 인덱싱/캐시 준비 중..."):
        vs, pages = load_or_build_index(uploaded.read(), chunk_size, chunk_overlap, OPENAI_API_KEY)
        st.session_state.vs = vs
        st.session_state.pages = pages
        st.session_state.uploaded_name = uploaded.name
        st.session_state.digest_df = None
    st.success("인덱싱 완료!")

# =========================
# 일반 체인 빌더 (질문 우선·중립 프롬프트)
# =========================
def build_chain(retriever):
    system = (
        "너는 업로드된 PDF에 근거해 한국어로 답한다. "
        "항상 **사용자 질문의 의도와 개수 요구를 정확히 따른다**. "
        "가능하면 bullet을 사용하고, 각 핵심 주장 끝에 [p.페이지] 근거를 붙인다. "
        "문서와 무관한 추측은 금지한다."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system + "\n\nCONTEXT:\n{context}"), ("human", "{question}")]
    )
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_chat(max_tokens=max_tokens, temperature=temperature)
        | StrOutputParser()
    )
    return chain

def render_strategy_md(data:dict) -> str:
    if not isinstance(data, dict): return str(data)
    def _arr(a): return a if isinstance(a, list) else []
    buf = []
    if data.get("summary"): buf += ["### 개요", data["summary"], ""]
    if data.get("north_star"): buf += [f"**North Star:** {data['north_star']}", ""]
    if _arr(data.get("assumptions")):
        buf += ["### 주요 가정", *[f"- {x}" for x in data["assumptions"]], ""]
    if _arr(data.get("strategies")):
        buf += ["### 핵심 전략"]
        for i, s in enumerate(data["strategies"], 1):
            buf += [f"#### {i}. {s.get('name','전략')}",
                    f"- **근거**: {s.get('rationale','')}",
                    f"- **임팩트 가설**: {s.get('expected_impact',{})}",
                    "- **실행(90d)**: " + ", ".join(_arr(s.get('actions',{}).get('90d',[]))),
                    "- **실행(180d)**: " + ", ".join(_arr(s.get('actions',{}).get('180d',[]))),
                    "- **실행(12m)**: " + ", ".join(_arr(s.get('actions',{}).get('12m',[]))),
                    "- **KPI**: " + ", ".join(_arr(s.get('kpis'))),
                    "- **리스크/대응**: " + "; ".join([f"{r.get('risk')}: {r.get('mitigation')}" for r in _arr(s.get('risks'))]),
                    "- **근거**: " + ", ".join(_arr(s.get('evidence'))),
                    ""]
    if _arr(data.get("data_needs")):
        buf += ["### 추가 데이터/실험 필요", *[f"- {x}" for x in data["data_needs"]]]
    return "\n".join(buf)


# =========================
# Q&A 라우팅 (의도별)
# =========================
if st.session_state.vs:
    st.markdown("---")
    st.markdown("### 💬 질문/답변")

    if st.button("질문하기") and question:
        with st.spinner("생각 중..."):
            intent = detect_intent(question)
            n = extract_count(question, 3)

            # 리트리버 (MMR + 압축)
            base_retriever = st.session_state.vs.as_retriever(
                search_type="mmr", search_kwargs={"k": top_k, "fetch_k": fetch_k}
            )
            retriever = base_retriever
            if use_compression:
                compressor = LLMChainExtractor.from_llm(
                    llm_light(max_tokens=300, temperature=0)  # 압축은 nano로 비용↓
                )
                retriever = ContextualCompressionRetriever(
                    base_retriever=base_retriever, base_compressor=compressor
                )
            
                        # === NEW: 컨설팅 모드 ===
            if strategy_mode:
                # 리트리버 준비 (MMR + 압축)
                base_retriever = st.session_state.vs.as_retriever(
                    search_type="mmr", search_kwargs={"k": top_k, "fetch_k": fetch_k}
                )
                retriever = base_retriever
                if use_compression:
                    compressor = LLMChainExtractor.from_llm(
                        llm_light(max_tokens=300, temperature=0)
                    )
                    retriever = ContextualCompressionRetriever(
                        base_retriever=base_retriever, base_compressor=compressor
                    )

                with st.spinner("요약 중..."):
                    brief = summarize_corpus(retriever, st.session_state.pages)

                with st.spinner("전략 분석 중..."):
                    strat = analyze_strategy(
                        brief=brief,
                        goal=strat_goal,
                        horizon=strat_horizon,
                        budget=strat_budget,
                        segment=strat_segment,
                        competitors=strat_competitors,
                    )
                    answer = render_strategy_md(strat)

                st.markdown("#### 🧠 답변")
                st.write(answer)
                # 인용 페이지 미리보기 유지
                cited_pages = extract_page_citations(answer)
                if cited_pages and st.session_state.pages:
                    with st.expander(f"🔎 인용된 페이지 스니펫 보기 ({len(cited_pages)}개)"):
                        page_map = {pg: txt for pg, txt in st.session_state.pages}
                        for pg in cited_pages:
                            snippet = (page_map.get(pg) or "")[:500]
                            st.markdown(f"**[p.{pg}]**\n\n{snippet}")
                st.stop()  # 컨설팅 모드에서는 여기서 종료


            if intent == "pros_cons":
                answer = answer_pros_cons(
                    question=question,
                    retriever=retriever,
                    pages=st.session_state.pages,
                    n=n,
                    use_global=enable_global_critique,
                    per_page_chars=global_critique_pages,
                    total_chars=global_critique_total,
                    api_key=OPENAI_API_KEY,
                    max_tokens=max_tokens + 250
                )
            elif intent == "critique" and enable_global_critique and st.session_state.pages:
                answer = critique_answer_global(
                    st.session_state.pages,
                    per_page_chars=global_critique_pages,
                    total_chars=global_critique_total,
                    api_key=OPENAI_API_KEY,
                    question=question,
                    max_tokens=max_tokens + 150
                )
            else:
                chain = build_chain(retriever)
                answer = chain.invoke(question)

        st.markdown("#### 🧠 답변")
        # 빈 문자열 가드
        if not (answer or "").strip():
            answer = "⚠️ 응답이 비어 있습니다. 다시 시도하거나 토큰 한도를 낮춰보세요."
        st.write(answer)

        # 근거 스니펫 미리보기
        cited_pages = extract_page_citations(answer)
        if cited_pages and st.session_state.pages:
            with st.expander(f"🔎 인용된 페이지 스니펫 보기 ({len(cited_pages)}개)"):
                page_map = {pg: txt for pg, txt in st.session_state.pages}
                for pg in cited_pages:
                    snippet = (page_map.get(pg) or "")[:500]
                    st.markdown(f"**[p.{pg}]**\n\n{snippet}")
else:
    st.info("PDF를 업로드하면 질문할 수 있습니다.")
