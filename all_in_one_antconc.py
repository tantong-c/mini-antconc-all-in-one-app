import streamlit as st
import pandas as pd
from collections import Counter
import re
import io

# 📌 นำเข้าไลบรารีตัดคำ (PyThaiNLP)
from pythainlp import word_tokenize

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Mini AntConc All-in-One",
                   layout="wide", page_icon="🐜")

# --- 🎨 Theme Settings ---
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    [data-testid="stSidebar"] { background-color: #E9ECEF; }
    th { background-color: #DEE2E6 !important; color: #212529 !important; }
    .search-help { background-color: #ffffff; padding: 15px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 🛠️ ฟังก์ชันประมวลผล (แก้ไข: กรอง | และช่องว่างทิ้งแน่นอน) ---


@st.cache_data(show_spinner=False)
def process_corpus(uploaded_files, auto_tokenize=True):
    """
    อ่านไฟล์ -> (ตัดคำ) -> กรองช่องว่างและเครื่องหมาย | ทิ้ง -> เตรียมข้อมูล
    """
    files_data = []
    all_tokens_flat = []

    for uploaded_file in uploaded_files:
        string_data = uploaded_file.read().decode("utf-8")

        tokens = []
        display_text = ""

        if auto_tokenize:
            # ✅ กรณี 1: Auto Tokenize
            raw_tokens = word_tokenize(
                string_data, engine='newmm', keep_whitespace=False)
        else:
            # ✅ กรณี 2: Manual (มี | มาแล้ว)
            # แทนที่ Newline ด้วย | ก่อน แล้วค่อย split
            clean_text = string_data.replace("\n", "|")
            raw_tokens = clean_text.split("|")

        # 🧹 Cleaning Step (หัวใจสำคัญ):
        # 1. strip() : ลบช่องว่างหน้าหลัง
        # 2. if t.strip() != "" : ไม่เอาคำว่าง
        # 3. if t.strip() != "|" : ไม่เอาเครื่องหมาย pipe (สำคัญ!)
        tokens = [
            t.strip() for t in raw_tokens
            if t.strip() != "" and t.strip() != "|"
        ]

        # สร้าง text สำหรับแสดงผล (File Content)
        display_text = "|".join(tokens)

        # เก็บข้อมูล
        files_data.append({
            "filename": uploaded_file.name,
            "tokens": tokens,
            "text": display_text
        })
        all_tokens_flat.extend(tokens)

    return files_data, all_tokens_flat

# --- ฟังก์ชัน Search Helper ---


def check_token_match(token, pattern):
    token = token.lower()
    pattern = pattern.lower()

    # 4. *คำ* (Strict Contains)
    if pattern.startswith("*") and pattern.endswith("*") and len(pattern) > 2:
        clean_pat = pattern[1:-1]
        return clean_pat in token[1:-1]

    # 2. *คำ (Strict Ends with)
    elif pattern.startswith("*") and len(pattern) > 1:
        clean_pat = pattern[1:]
        return token.endswith(clean_pat) and len(token) > len(clean_pat)

    # 3. คำ* (Strict Starts with)
    elif pattern.endswith("*") and len(pattern) > 1:
        clean_pat = pattern[:-1]
        return token.startswith(clean_pat) and len(token) > len(clean_pat)

    # 1. คำ (Exact match)
    else:
        return token == pattern


def parse_search_query(query):
    query = query.strip()
    gap_pattern = re.search(r'^(\S+)\s+<(\d+)(?:-(\d+))?>\s+(\S+)$', query)
    sequence_pattern = re.search(r'^(\S+)\s+(\S+)$', query)

    if gap_pattern:
        start_word = gap_pattern.group(1)
        min_gap = int(gap_pattern.group(2))
        max_gap = int(gap_pattern.group(
            3)) if gap_pattern.group(3) else min_gap
        end_word = gap_pattern.group(4)
        return "gap", (start_word, min_gap, max_gap, end_word)
    elif sequence_pattern:
        return "gap", (sequence_pattern.group(1), 0, 0, sequence_pattern.group(2))
    else:
        return "single", query

# --- ฟังก์ชัน Analysis Core ---


def generate_kwic(files_data, keyword, window_size=7):
    results = []
    search_type, search_params = parse_search_query(keyword)

    for file_info in files_data:
        filename = file_info['filename']
        tokens = file_info['tokens']
        len_tokens = len(tokens)
        i = 0
        while i < len_tokens:
            match_found = False
            match_start = i
            match_end = i

            if search_type == "single":
                if check_token_match(tokens[i], search_params):
                    match_found = True
                    match_end = i
            elif search_type == "gap":
                start_pat, min_gap, max_gap, end_pat = search_params
                if check_token_match(tokens[i], start_pat):
                    s_range = i + 1 + min_gap
                    e_range = min(i + 1 + max_gap + 1, len_tokens)
                    for j in range(s_range, e_range):
                        if check_token_match(tokens[j], end_pat):
                            match_found = True
                            match_end = j
                            break

            if match_found:
                left = tokens[max(0, match_start - window_size):match_start]
                node = tokens[match_start:match_end+1]
                right = tokens[match_end +
                               1:min(len_tokens, match_end + window_size + 1)]
                results.append({
                    "Left": " ".join(left),
                    "Node": " ".join(node),
                    "Right": " ".join(right),
                    "File": filename
                })
            i += 1
    return pd.DataFrame(results)


def generate_ngrams(tokens, n=2, min_freq=1):
    if len(tokens) < n:
        return pd.DataFrame()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    counts = Counter([" ".join(ngram) for ngram in ngrams])
    df = pd.DataFrame(counts.items(), columns=['Cluster', 'Frequency'])
    return df[df['Frequency'] >= min_freq].sort_values(by='Frequency', ascending=False).reset_index(drop=True)


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 🖥️ UI Section ---


st.title("🐜 Mini AntConc All-in-One")
st.caption("อัปโหลดไฟล์ดิบ -> ตัดคำอัตโนมัติ -> วิเคราะห์ผล")
st.caption("Upload Raw Files -> Auto-segment -> Analyze Results")

with st.sidebar:
    st.header("📂 Upload & Settings")

    use_auto_tokenize = st.checkbox("ให้โปรแกรมตัดคำให้ (Auto Tokenize)",
                                    value=True, help="เลือกอันนี้ถ้าไฟล์คุณเป็นข้อความธรรมดาที่ยังไม่ได้ใส่ |")

    uploaded_files = st.file_uploader(
        "เลือกไฟล์ Text (UTF-8)", type=['txt'], accept_multiple_files=True)

    if use_auto_tokenize:
        st.info("ℹ️ กำลังใช้ระบบตัดคำ: PyThaiNLP (newmm)")
    else:
        st.warning("⚠️ โหมด Manual: ไฟล์ต้องมี | คั่นคำมาแล้ว")

    st.link_button("Thai Word Segmenter App",
                   "https://thai-word-seg-app-rhvzfn7jkxytwlqydi8idq.streamlit.app/")

if uploaded_files:
    with st.spinner('กำลังอ่านไฟล์และตัดคำ (Tokenizing)...'):
        files_data, all_tokens_flat = process_corpus(
            uploaded_files, auto_tokenize=use_auto_tokenize)

    st.success(
        f"✅ ประมวลผลเสร็จสิ้น! {len(files_data)} ไฟล์ | {len(all_tokens_flat):,} คำ")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Concordance", "📊 Word List", "🔗 N-Grams", "📄 File Content"])

    # --- Tab 1: KWIC ---
    with tab1:
        st.subheader("Concordance Tool")
        with st.expander("ℹ️ วิธีการใช้คำค้นหา (Search Syntax Guidelines) - คลิกเพื่ออ่าน"):
            st.markdown("""
            **คุณสามารถใช้รูปแบบคำค้นหาได้ดังนี้:**
            
            1.  **รัก** : ค้นหาคำว่า "รัก" เท่านั้น (Exact match)
            2.  **\*รัก** : ค้นหาคำที่ **ลงท้าย** ด้วย "รัก" (เช่น น่ารัก, ความรัก)
            3.  **รัก\*** : ค้นหาคำที่ **ขึ้นต้น** ด้วย "รัก" (เช่น รักษา, รักใคร่)
            4.  **\*รัก\*** : ค้นหาคำที่มี "รัก" **อยู่ในคำ** (เช่น อนุรักษ์)
            5.  **รัก มาก** : ค้นหาคำว่า "รัก" ตามด้วย "มาก"
            6.  **รัก <3> มาก** : "รัก" ตามด้วยคำอื่น **3 คำ** แล้วตามด้วย "มาก"
            7.  **รัก <0-3> มาก** : "รัก" ตามด้วยคำอื่น **0 ถึง 3 คำ** แล้วตามด้วย "มาก"
            """)

        c1, c2 = st.columns([3, 1])
        search_term = c1.text_input("คำค้นหา KWIC (Keyword in Context):", "")
        window = c2.slider("บริบท (Context Span):", 3, 20, 8)

        if search_term:
            df = generate_kwic(files_data, search_term, window)
            if not df.empty:
                df.index += 1
                st.write(f"พบ: {len(df)} รายการ")
                st.download_button("📥 CSV", convert_df_to_csv(
                    df), f"kwic_{search_term}.csv", "text/csv")
                st.dataframe(df, use_container_width=True, column_config={"Left": st.column_config.TextColumn(
                    width="medium"), "Node": st.column_config.TextColumn(width="small"), "Right": st.column_config.TextColumn(width="medium")})
            else:
                st.warning("ไม่พบคำที่ค้นหา")

    # --- Tab 2: Word List ---
    with tab2:
        st.subheader("Word List")
        wc = Counter(all_tokens_flat)
        df_wl = pd.DataFrame(wc.items(), columns=['Word', 'Frequency']).sort_values(
            'Frequency', ascending=False).reset_index(drop=True)
        df_wl.index += 1
        st.write(f"พบ: {len(wc)} รายการ")
        st.download_button("📥 CSV", convert_df_to_csv(
            df_wl), "wordlist.csv", "text/csv")
        st.dataframe(df_wl, use_container_width=True)

    # --- Tab 3: N-Grams ---
    with tab3:
        st.subheader("N-Grams")
        c1, c2 = st.columns(2)
        n_size = c1.number_input("N-gram size", 2, 5, 2)
        min_f = c2.number_input("Min Frequency", 1, 100, 2)

        if st.button("Start N-Grams"):
            df_ng = generate_ngrams(all_tokens_flat, n_size, min_f)
            if not df_ng.empty:
                df_ng.index += 1
                st.write(f"พบ: {len(df_ng)} รายการ")
                st.download_button("📥 CSV", convert_df_to_csv(
                    df_ng), f"{n_size}grams.csv", "text/csv")
                st.dataframe(df_ng, use_container_width=True)
            else:
                st.warning("ไม่พบข้อมูล")

    # --- Tab 4: Content ---
    with tab4:
        st.subheader("เนื้อหาที่ผ่านการตัดคำแล้ว")
        sel_f = st.selectbox("เลือกไฟล์:", [f['filename'] for f in files_data])
        txt = next((i['text']
                   for i in files_data if i['filename'] == sel_f), "")
        st.text_area("Content (Tokenized view):", txt, height=400)
        st.caption("*เครื่องหมาย | แสดงจุดที่โปรแกรมทำการตัดคำ")

else:
    st.info("👈 กรุณาเลือกไฟล์ Text ทางด้านซ้ายเพื่อเริ่มใช้งาน")
