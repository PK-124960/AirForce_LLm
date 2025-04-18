
import os
import re
from PIL import Image
import streamlit as st

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="🛠️ Afterburner Assistant", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }

    .stTextInput > label {
        font-weight: bold;
        color: #2D2D2D;
        font-size: 1rem;
        margin-bottom: 5px;
    }

    input[type="text"] {
        border: 2px solid #3A86FF !important;
        padding: 10px;
        border-radius: 8px;
        background-color: #ffffff;
        font-size: 1rem;
    }

    input[type="text"]:focus {
        border-color: #FF006E !important;
        box-shadow: 0 0 0 0.2rem rgba(255,0,110,.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- FUNCTION ----------
@st.cache_resource
def load_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = FAISS.load_local("faiss_afterburner_index", embedding, allow_dangerous_deserialization=True)
    llm = Ollama(model="mistral")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return chain, vectordb

def extract_figures_from_text(text):
    matches = re.findall(r"Figure\s*(\d+)[\.-](\d+)", text, flags=re.IGNORECASE)
    filenames = []
    for a, b in matches:
        base = f"figure_{int(a)}_{int(b)}"
        for i in range(1, 4):  # เผื่อมีหลายหน้า
            multi = os.path.join("figures", f"{base}_{i}.png")
            if os.path.exists(multi):
                filenames.append(multi)
            elif i == 1:
                fallback = os.path.join("figures", f"{base}.png")
                if os.path.exists(fallback):
                    filenames.append(fallback)
                    break
    return filenames



def show_figure_images(image_paths):
    for path in image_paths:
        st.image(Image.open(path), caption=os.path.basename(path).replace("_", " ").replace(".png", ""), use_column_width=True)

# ---------- LOAD CHAIN ----------
qa_chain, vectordb = load_chain()

# ---------- UI ----------
st.title("🛠️ Afterburner Q&A Assistant (Prototype)")
question = st.text_input("ถามคำถามเกี่ยวกับคู่มือ Afterburner เช่น: การถอด actuating cylinder ต้องทำอย่างไร?")

if question:
    with st.spinner("🔍 กำลังค้นหาและตอบกลับ..."):
        # ดึงเนื้อหาจาก FAISS
        docs = vectordb.similarity_search(question, k=1)
        page_number = docs[0].metadata.get("page", None)

        # ใช้ LLM ตอบ
        answer = qa_chain.run(
            f"คุณคือผู้ช่วยด้านเทคนิคภาษาไทย ช่วยตอบคำถามนี้โดยย่อ:\n\"{question}\""
        )


        # แสดงผล
        st.success("📌 คำตอบ:")
        st.markdown(answer)

        # ตรวจหาและแสดงภาพ Figure
        fig_paths = extract_figures_from_text(answer)
        if fig_paths:
            st.markdown("🖼️ **ภาพประกอบจาก Figure:**")
            show_figure_images(fig_paths)

        # แสดงแหล่งที่มาเป็นหน้า (fallback)
        if page_number:
            st.markdown(f"📖 **แหล่งที่มาโดยประมาณ: หน้า {page_number}**")
