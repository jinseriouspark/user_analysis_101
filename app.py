import os
import re
import json
import platform
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

import google.generativeai as genai
from notion_client import Client

# =========================
# 기본 UI 설정
# =========================
st.set_page_config(page_title="온톨로지 기반 프롬프트 추출", layout="wide")
st.title("🧑 온톨로지 기반 Persona Prompt Builder")

# 브라우저/Plotly 폰트 (Streamlit Cloud에서는 packages.txt에 fonts-nanum 권장)
PLOTLY_FONT = "Nanum Gothic, Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
st.markdown(f"""
<style>
html, body, [class*="css"] {{
  font-family: {PLOTLY_FONT};
}}
</style>
""", unsafe_allow_html=True)

# Matplotlib 한글 세팅
def set_korean_font():
    try:
        system = platform.system()
        if system == "Darwin":
            plt.rc("font", family="AppleGothic")
        elif system == "Windows":
            plt.rc("font", family="Malgun Gothic")
        else:
            # Linux: Nanum 설치 시 자동 인식
            pass
        plt.rc("axes", unicode_minus=False)
    except Exception:
        pass

set_korean_font()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### 🔧 환경 설정")
    NOTION_TOKEN = st.text_input("Notion API Token", type="password")
    NOTION_DB_ID = st.text_input("원본 DB ID (32자리)", help="URL 중간의 32자 Database ID")
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")
    MODEL_NAME = st.selectbox(
        "Gemini 모델",
        ["gemini-2.0-flash", "gemini-2.5-pro"],
        index=0,
        help="⚡ gemini-2.0-flash: 빠르고 토큰 효율적 / 🎯 gemini-2.5-pro: 고퀄리티 답변"
    )
    st.caption("⚡ flash = 효율 / 🎯 pro = 고퀄리티")

# =========================
# 세션 상태
# =========================
if "df_rows" not in st.session_state:
    st.session_state.df_rows = None          # 원자료
if "grouped" not in st.session_state:
    st.session_state.grouped = None          # 이름별 합본
if "analysis" not in st.session_state:
    st.session_state.analysis: Dict[str, Dict[str, Any]] = {}  # 캐시

# =========================
# Notion: 행 수집
# =========================
def fetch_notion_rows(notion: Client, db_id: str) -> List[Dict[str, Any]]:
    rows = []
    cursor = None
    while True:
        resp = notion.databases.query(database_id=db_id, start_cursor=cursor) if cursor \
               else notion.databases.query(database_id=db_id)
        for row in resp.get("results", []):
            props = row.get("properties", {})
            # 이름
            name = ""
            try:
                title_items = props.get("이름", {}).get("title", [])
                if title_items:
                    name = title_items[0].get("plain_text") or title_items[0].get("text", {}).get("content", "")
            except Exception:
                pass
            # 텍스트
            text = ""
            try:
                rich = props.get("텍스트", {}).get("rich_text", [])
                if rich:
                    text = rich[0].get("plain_text") or rich[0].get("text", {}).get("content", "")
            except Exception:
                pass
            rows.append({"이름": name or "이름없음", "텍스트": text or ""})
        if resp.get("has_more"):
            cursor = resp.get("next_cursor")
        else:
            break
    return rows

# =========================
# Gemini 유틸
# =========================
def parse_json_safely(text: str) -> Dict[str, Any]:
    """모델 출력에서 JSON만 안전 추출"""
    if not text:
        return {}
    m = re.search(r"```json(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1).strip()
    m2 = re.search(r"\{.*\}", text, flags=re.S)
    if m2:
        text = m2.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}

def request_ontology_prompt(all_text: str) -> str:
    return f"""
다음 텍스트를 분석하여 **온톨로지(지식그래프)** 를 JSON으로 산출하라.

[출력 요구]
1) "themes": 문자열 배열 (정확히 5개)
2) "keywords": 문자열 배열 (최소 10, 최대 50)
3) "relationships": 배열 (원소는 {{ "source": 단어, "target": 단어, "relation": 관계설명 }})

[예시]
{{
  "themes": ["건강","자기관리","성취감","도전","실패"],
  "keywords": ["러닝","습관","목표","체력","멘탈","도전","실패","성장","꾸준함","회복력"],
  "relationships": [
    {{"source":"건강","target":"자기관리","relation":"자기 관리를 통해 건강을 지킨다"}},
    {{"source":"도전","target":"성취감","relation":"도전을 통해 성취감을 얻는다"}}
  ]
}}

[분석 텍스트]
{all_text}
"""

def request_summary_prompt(all_text: str) -> str:
    return f"""
아래 텍스트를 바탕으로, 이 사람의 **과거 경험·관심사·반복 주제·말투**를 한국어로 간결하게 요약하라.
- 불필요한 일반론 금지, 텍스트 내용에 근거할 것.
- 5~10문장 이내.

텍스트:
{all_text}
"""

def stream_generate_text(model, prompt: str, lang_hint: str = "json") -> str:
    """
    Gemini 스트리밍 응답을 실시간 표시하고 최종 텍스트를 반환.
    lang_hint: "json", "markdown" 등 코드 하이라이트용.
    """
    title_ph = st.empty()
    code_ph = st.empty()
    buffer = ""
    title_ph.markdown("**📡 실시간 출력(원문)** — 생성 중…")
    try:
        resp = model.generate_content(prompt, stream=True)
        for chunk in resp:
            t = getattr(chunk, "text", "") or ""
            if not t:
                continue
            buffer += t
            code_ph.code(buffer, language=lang_hint)
        title_ph.markdown("**📡 실시간 출력(원문)** — ✅ 수신 완료")
    except Exception as e:
        title_ph.markdown("**📡 실시간 출력(원문)** — ❌ 오류")
        code_ph.code(f"Streaming error: {e}", language="text")
    return buffer

# =========================
# 최종 프롬프트 생성
# =========================
def build_final_prompt(name: str, ontology: Dict[str, Any], summary: str) -> str:
    themes_list = ", ".join(ontology.get("themes", []))
    keywords_list = ", ".join(ontology.get("keywords", []))
    rel_lines = []
    for r in (ontology.get("relationships") or []):
        s = (r.get("source") or "").strip()
        t = (r.get("target") or "").strip()
        rel = (r.get("relation") or "").strip()
        if s and t:
            rel_lines.append(f"- {s} ↔ {t} : {rel}" if rel else f"- {s} ↔ {t}")
    relationships_bullets = "\n".join(rel_lines) if rel_lines else "- (관계 없음)"

    return f"""당신은 이제부터 **{name}** 이며, 컨설팅을 받으러 온 **고객**입니다.
당신은 AI 어시스턴트가 아닙니다. 시뮬레이션 대화에 고객 역할로 참여합니다.
**모든 발화는 반드시 한국어로만** 하세요.

[퍼소나 & 근거자료]
- {name}의 말투/관심사/표현을 유지하세요.
- 아래 온톨로지와 과거 요약에 **반드시 기반**하여 말하세요.
- 근거 밖의 내용은 절대 추정하지 말고 “나는 잘 모른다.”라고 답하세요.

[온톨로지(Themes/Keywords/Relations)]
- 주제(5): {themes_list}
- 키워드(10~50): {keywords_list}
- 관계(에지):
{relationships_bullets}

[과거 요약]
{summary}

[대화 목표]
- 당신(고객)은 **자신의 사업**에 대한 컨설팅을 받으러 왔습니다.
- 당신의 임무는 **좋은 질문을 많이 던지는 것**이며, 필요할 때만 간결한 맥락을 제공합니다.

[대화 스타일 규칙 (한국어 전용)]
1) **질문 중심**: 거의 매 턴마다 내 사업의 목표·지표·고객·제약·타임라인·리스크에 대해 **핵심 질문 1~2개**를 던지세요.
2) **짧고 구체적으로**: 본인의 설명은 2~5문장 이내로 간결하게. 장문의 강의/설명 금지.
3) **퍼소나 유지**: {name}의 과거 요약과 온톨로지에 맞는 어휘/톤을 사용.
4) **근거만 사용**: 온톨로지/과거 요약에 없는 사실은 말하지 마세요. 모르면 “나는 잘 모른다.”
5) **티칭 모드 금지**: 프레임워크 강의/일방적 조언 대신 **묻고 확인**하는 대화로 이끕니다.
6) **점진적 파고들기**: 넓게 시작 → 빠르게 좁혀서 (시장/고객/가설/수치/예산/조직/리스크)로 파고드세요.
7) **역할 유지**: 시뮬레이션을 절대 깨지 말고, AI/시뮬레이션 언급 금지.

[오프닝 턴 가이드]
- 먼저 현재 상황을 **1~2문장**으로 요약하세요(온톨로지/과거 요약에 근거).
- 이어서 **핵심 질문 1~2개**를 던지세요.
"""

# =========================
# 온톨로지 3D 시각화
# =========================
def plot_ontology_3d(ontology: Dict[str, Any], title: str):
    edges = ontology.get("relationships", []) or []
    if not edges:
        st.info("관계(에지)가 없어 그래프를 그릴 수 없습니다.")
        return

    G = nx.Graph()
    for rel in edges:
        s = rel.get("source"); t = rel.get("target"); lbl = rel.get("relation", "")
        if s and t:
            G.add_edge(s, t, label=lbl)

    pos = nx.spring_layout(G, dim=3, seed=42)
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    z_nodes = [pos[n][2] for n in G.nodes()]

    edge_x, edge_y, edge_z = [], [], []
    for e in G.edges():
        x0, y0, z0 = pos[e[0]]
        x1, y1, z1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode="markers+text",
        marker=dict(size=8),
        text=list(G.nodes()),
        textposition="top center"
    ))
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(width=2)
    ))
    fig.update_layout(title=title, showlegend=False, height=520,
                      font=dict(family=PLOTLY_FONT))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 분석(스트리밍) + 렌더
# =========================
def analyze_and_render_streaming(name: str, all_text: str, model):
    # 1) 온톨로지 스트리밍
    st.markdown("#### 📡 온톨로지 생성 스트림")
    raw_ontology_text = stream_generate_text(model, request_ontology_prompt(all_text), lang_hint="json")
    ontology = parse_json_safely(raw_ontology_text)
    if not ontology:
        st.error("온톨로지 JSON 파싱 실패 (스트림 원문을 확인하세요).")
        return

    st.markdown("#### 📌 온톨로지 (JSON)")
    st.code(json.dumps(ontology, ensure_ascii=False, indent=2), language="json")

    if ontology.get("relationships"):
        st.markdown("#### 📌 온톨로지 (관계 테이블)")
        st.dataframe(pd.DataFrame(ontology["relationships"]), use_container_width=True)

        st.markdown("#### 📌 온톨로지 3D 그래프")
        plot_ontology_3d(ontology, f"{name} 온톨로지 그래프")

    # 2) 요약 스트리밍
    st.markdown("#### 📡 과거 요약 생성 스트림")
    raw_summary_text = stream_generate_text(model, request_summary_prompt(all_text), lang_hint="markdown")
    summary = raw_summary_text.strip()

    st.markdown("#### 📌 과거 요약")
    st.write(summary)

    # 3) 최종 프롬프트
    final_prompt = build_final_prompt(name, ontology, summary)
    st.markdown("#### 🧾 최종 프롬프트 (한국어·고객역할)")
    st.code(final_prompt, language="markdown")

    # 다운로드
    st.download_button(
        label="⬇️ 프롬프트 저장 (.txt)",
        data=final_prompt.encode("utf-8"),
        file_name=f"{name}_final_prompt.txt",
        mime="text/plain",
        key=f"dl_{name}"
    )

    # 캐시 저장
    st.session_state.analysis[name] = {
        "ontology": ontology,
        "summary": summary,
        "final_prompt": final_prompt,
    }

# =========================
# 본문 UI
# =========================
st.markdown("### 1) Notion DB 불러오기")
if st.button("📥 Notion DB 불러오기", type="primary"):
    if not (NOTION_TOKEN and NOTION_DB_ID):
        st.error("Notion Token과 원본 DB ID를 입력하세요.")
    else:
        try:
            notion = Client(auth=NOTION_TOKEN)
            rows = fetch_notion_rows(notion, NOTION_DB_ID)
            st.session_state.df_rows = pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Notion 불러오기 실패: {e}")

df = st.session_state.df_rows
if df is not None and not df.empty:
    st.dataframe(df, use_container_width=True, height=360)
    st.markdown("---")

    st.markdown("### 2) 이름 기준 그룹화 (유저별 전체 텍스트 합치기)")
    grouped = df.groupby("이름")["텍스트"].apply(lambda s: "\n\n".join([x for x in s if x])).reset_index()
    st.session_state.grouped = grouped

    prev = grouped.copy()
    prev["텍스트 샘플"] = prev["텍스트"].str.slice(0, 80) + "..."
    st.dataframe(prev[["이름", "텍스트 샘플"]], use_container_width=True, height=280)
    st.markdown("---")

    st.markdown("### 3) 유저별 분석 (버튼 누르면 실행)")
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name=MODEL_NAME)
        except Exception as e:
            model = None
            st.error(f"Gemini 모델 초기화 실패: {e}")
    else:
        model = None

    for idx, r in grouped.iterrows():
        name = r["이름"]; all_text = r["텍스트"]
        with st.expander(f"👤 {name}", expanded=False):
            st.caption("※ 버튼을 누르기 전까지는 분석하지 않습니다.")
            if st.button("🧪 프롬프트 추출", key=f"extract_{idx}"):
                if not model:
                    st.error("Gemini API Key를 입력하거나 모델을 다시 선택하세요.")
                elif name in st.session_state.analysis:
                    st.info("이미 분석된 결과가 있어 캐시를 사용합니다. 재분석하려면 페이지 새로고침.")
                    cached = st.session_state.analysis[name]
                    st.markdown("#### 📌 온톨로지 (JSON)")
                    st.code(json.dumps(cached["ontology"], ensure_ascii=False, indent=2), language="json")

                    if cached["ontology"].get("relationships"):
                        st.markdown("#### 📌 온톨로지 (관계 테이블)")
                        st.dataframe(pd.DataFrame(cached["ontology"]["relationships"]), use_container_width=True)
                        st.markdown("#### 📌 온톨로지 3D 그래프")
                        plot_ontology_3d(cached["ontology"], f"{name} 온톨로지 그래프")

                    st.markdown("#### 📌 과거 요약")
                    st.write(cached["summary"])

                    st.markdown("#### 🧾 최종 프롬프트 (한국어·고객역할)")
                    st.code(cached["final_prompt"], language="markdown")
                    st.download_button(
                        label="⬇️ 프롬프트 저장 (.txt)",
                        data=cached["final_prompt"].encode("utf-8"),
                        file_name=f"{name}_final_prompt.txt",
                        mime="text/plain",
                        key=f"dl_cached_{name}"
                    )
                else:
                    analyze_and_render_streaming(name, all_text, model)
else:
    st.info("사이드바에 키를 입력하고, 먼저 **📥 Notion DB 불러오기**를 눌러주세요.")
