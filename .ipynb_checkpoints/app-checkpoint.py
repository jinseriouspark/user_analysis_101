import os
import platform
import streamlit as st
import google.generativeai as genai
from notion_client import Client
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ----------------------
# 자동 한글 폰트 감지
# ----------------------
def set_korean_font():
    system = platform.system()
    if system == "Darwin":  # macOS
        plt.rc('font', family='AppleGothic')
    elif system == "Windows":
        plt.rc('font', family='Malgun Gothic')
    else:  # Linux (Colab, Ubuntu 등)
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(font_path):
            plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        else:
            print("⚠️ NanumGothic not found. Run: sudo apt-get install -y fonts-nanum")

    # 마이너스 깨짐 방지
    plt.rc('axes', unicode_minus=False)

set_korean_font()

# ----------------------
# Streamlit 기본 설정
# ----------------------
st.set_page_config(page_title="유저별 온톨로지 분석", layout="wide")
st.title("🧑 유저별 온톨로지 분석 & 프롬프트 생성기")

# ----------------------
# 입력값
# ----------------------
NOTION_TOKEN = st.text_input("🔑 Notion API Token", type="password")
NOTION_DB_ID = st.text_input("📂 Notion DB ID (주소에서 추출)")
GEMINI_API_KEY = st.text_input("🤖 Gemini API Key", type="password")

# ----------------------
# 분석 버튼
# ----------------------
if st.button("🚀 분석하기"):
    if not (NOTION_TOKEN and NOTION_DB_ID and GEMINI_API_KEY):
        st.error("⚠️ 모든 입력값을 채워주세요.")
    else:
        # 1. 초기화
        notion = Client(auth=NOTION_TOKEN)
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # 2. Notion DB 데이터 가져오기
        results = notion.databases.query(database_id=NOTION_DB_ID)
        rows = []
        for row in results["results"]:
            name = row["properties"]["이름"]["title"][0]["plain_text"]
            text = ""
            if row["properties"]["텍스트"]["rich_text"]:
                text = row["properties"]["텍스트"]["rich_text"][0]["plain_text"]
            rows.append({"name": name, "text": text})

        # 3. 각 유저별 분석
        for row in rows:
            with st.expander(f"👤 {row['name']} 분석 결과"):
                text = row["text"]

                # 3-1. 온톨로지 생성
                ontology_prompt = f"""
                텍스트를 분석해 온톨로지를 만들어라.
                - 핵심 개념 단어를 5개 이내로 추출
                - 단어 간의 관계를 'X ↔ Y : 관계설명' 형태로 정리
                텍스트: {text}
                """
                ontology = model.generate_content(ontology_prompt).text

                # 3-2. 요약
                summary_prompt = f"""
                다음 텍스트를 요약하라.
                - 핵심 주제, 말투, 자주 쓰는 단어, 관심사를 포함할 것.
                텍스트: {text}
                """
                summary = model.generate_content(summary_prompt).text

                # 3-3. 최종 프롬프트
                final_prompt = f"""
                You are going to roleplay as {row['name']}.

                [Ontology]
                {ontology}

                [Summary of Writing Style]
                {summary}

                [Rules]
                - Always answer as if you are {row['name']}.
                - Use only the ontology and summary above.
                - If a question is outside the scope, say "나는 잘 모른다."
                - Do not break roleplay. Never mention AI.
                """

                # ----------------------
                # 결과 출력
                # ----------------------
                st.markdown("### 📌 온톨로지")
                st.write(ontology)

                st.markdown("### 📌 요약")
                st.write(summary)

                st.markdown("### 📌 최종 프롬프트")
                st.code(final_prompt, language="markdown")

                # ----------------------
                # 온톨로지 3D 그래프
                # ----------------------
                edges = []
                for line in ontology.split("\n"):
                    if "↔" in line:
                        parts = line.split(":")
                        relation = parts[1].strip() if len(parts) > 1 else ""
                        nodes = parts[0].split("↔")
                        if len(nodes) == 2:
                            src, dst = nodes[0].strip(), nodes[1].strip()
                            edges.append((src, dst, relation))

                if edges:
                    G = nx.Graph()
                    for src, dst, rel in edges:
                        G.add_edge(src, dst, label=rel)

                    pos = nx.spring_layout(G, dim=3, seed=42)

                    x_nodes = [pos[n][0] for n in G.nodes()]
                    y_nodes = [pos[n][1] for n in G.nodes()]
                    z_nodes = [pos[n][2] for n in G.nodes()]

                    edge_x, edge_y, edge_z = [], [], []
                    for edge in G.edges():
                        x0, y0, z0 = pos[edge[0]]
                        x1, y1, z1 = pos[edge[1]]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]
                        edge_z += [z0, z1, None]

                    fig = go.Figure()

                    # 노드
                    fig.add_trace(go.Scatter3d(
                        x=x_nodes, y=y_nodes, z=z_nodes,
                        mode='markers+text',
                        marker=dict(size=8, color='skyblue'),
                        text=list(G.nodes()),
                        textposition="top center"
                    ))

                    # 엣지
                    fig.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='gray', width=2)
                    ))

                    fig.update_layout(
                        title=f"{row['name']} 온톨로지 그래프",
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)
