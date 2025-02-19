import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# โหลดข้อมูล
@st.cache_data
def load_data():
    file_path = "Lineman_Shops_Final_Clean.csv"  # เปลี่ยนเป็น path ของคุณ
    df = pd.read_csv(file_path)
    df["combined_features"] = df["category"] + " " + df["cuisine"] + " " + df["price_level"]
    return df

df = load_data()

# สร้างโมเดล TF-IDF + Nearest Neighbors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

nn_model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="auto")
nn_model.fit(tfidf_matrix)

# ✅ ฟังก์ชันแก้ไขลิงก์ URL
def format_url(name, url):
    if pd.isna(url) or url.strip() == "-" or url.strip() == "":
        return f"https://www.google.com/search?q={name} ร้านอาหาร"  # ใช้ Google Search แทน
    return url

# ✅ ฟังก์ชันแนะนำร้านที่คล้ายกัน (แก้ไขแล้ว)
def recommend_similar_restaurants(restaurant_name, top_n=3):
    indices = df[df["name"].str.contains(restaurant_name, case=False, na=False)].index
    if len(indices) == 0:
        return ["❌ ไม่พบร้านที่คล้ายกัน"]

    idx = indices[0]
    distances, neighbors = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    restaurant_indices = neighbors[0][1:]

    results = []
    for i in restaurant_indices:
        row = df.iloc[i]
        paragraph = f"""
        🍽 **{row['name']}**  
        - **หมวดหมู่**: {row['category']}  
        - **ราคา**: {row['price_level']}  
        - **ที่อยู่**: {row['street']}  
        - 🔗 [ดูรายละเอียดเพิ่มเติม]({format_url(row['name'], row['url'])})  
        """
        results.append(paragraph)
    
    return results

# ✅ ฟังก์ชันแนะนำร้านตามตัวกรอง (แก้ไขแล้ว)
def recommend_restaurants(province, category, price_level, top_n=5):
    filtered_df = df[df["street"].str.contains(province, na=False, case=False)]
    filtered_df = filtered_df[filtered_df["category"].str.contains(category, na=False, case=False)]
    filtered_df = filtered_df[filtered_df["price_level"] == price_level]

    results = []
    for _, row in filtered_df.head(top_n).iterrows():
        paragraph = f"""
        🍽 **{row['name']}**  
        - **หมวดหมู่**: {row['category']}  
        - **ราคา**: {row['price_level']}  
        - **ที่อยู่**: {row['street']}  
        - 🔗 [ดูรายละเอียดเพิ่มเติม]({format_url(row['name'], row['url'])})  
        """
        results.append(paragraph)

    return results if results else ["❌ ไม่พบร้านที่ตรงกับเงื่อนไข"]

# ✅ สร้าง UI
st.title("🍽️ ระบบแนะนำร้านอาหาร - เย็นนี้กินอะไรดี?")
st.sidebar.header("🔍 ค้นหาร้านอาหาร")

# 🔹 ตัวเลือกค้นหาตามตัวกรอง
tab1, tab2 = st.tabs(["🔍 ค้นหาตามตัวกรอง", "🤖 แนะนำร้านที่คล้ายกัน"])

with tab1:
    province = st.selectbox("📍 เลือกจังหวัด", df["street"].dropna().unique())
    category = st.selectbox("🍜 เลือกประเภทอาหาร", df["category"].dropna().unique())
    price_level = st.selectbox("💰 เลือกระดับราคา", df["price_level"].unique())
    
    if st.button("🔍 ค้นหาร้านอาหาร"):
        results = recommend_restaurants(province, category, price_level)
        for res in results:
            st.markdown(res)

with tab2:
    restaurant_name = st.text_input("🏠 ใส่ชื่อร้านอาหารที่ต้องการหา")
    
    if st.button("🤖 แนะนำร้านที่คล้ายกัน"):
        similar_results = recommend_similar_restaurants(restaurant_name)
        for res in similar_results:
            st.markdown(res)

st.sidebar.markdown("### 📢 วิธีใช้")
st.sidebar.write("""
- ใช้ตัวกรองเพื่อค้นหาร้านอาหารตามจังหวัด, ประเภท และราคา
- หรือพิมพ์ชื่อร้านอาหารเพื่อหา "ร้านที่คล้ายกัน"
- คลิกที่ลิงก์ร้านเพื่อดูรายละเอียด
""")
