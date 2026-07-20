# ============================================================
# 🏥 Fitness Health Prediction App - Streamlit
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px



# เพิ่มโค้ดนี้ไว้บนสุดของ app.py (หลัง import)
import traceback

# ============================================================
# โหลดโมเดล - พร้อม Error Handling แบบละเอียด
# ============================================================
@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'svm_health_prediction_model.joblib')
        info_path = os.path.join(current_dir, 'streamlit_model_info.joblib')
        
        # ตรวจสอบว่ามีไฟล์หรือไม่
        if not os.path.exists(model_path):
            st.error(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
            st.info("📂 ไฟล์ในโฟลเดอร์ปัจจุบัน:")
            files = os.listdir(current_dir)
            for f in files:
                size = os.path.getsize(os.path.join(current_dir, f)) / 1024
                st.write(f"  • {f} ({size:.2f} KB)")
            return None, None
        
        if not os.path.exists(info_path):
            st.error(f"❌ ไม่พบไฟล์ metadata: {info_path}")
            return None, None
        
        model = joblib.load(model_path)
        info = joblib.load(info_path)
        return model, info
    
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล")
        st.error(f"**Error Type:** {type(e).__name__}")
        st.error(f"**Error Message:** {str(e)}")
        st.code(traceback.format_exc())
        return None, None












# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Fitness Health Predictor",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# โหลดโมเดลและ metadata
@st.cache_resource
def load_model():
    """โหลดโมเดลและ metadata"""
    model = joblib.load('svm_health_prediction_model.joblib')
    info = joblib.load('streamlit_model_info.joblib')
    return model, info

model, model_info = load_model()

# ============================================================
# Sidebar - ข้อมูลแอป
# ============================================================
with st.sidebar:
    st.title("🏋️ Fitness Health Predictor")
    st.markdown("---")
    st.markdown("""
    ### เกี่ยวกับแอปนี้
    แอปนี้ใช้โมเดล **Support Vector Machine (SVM)** 
    ในการทำนายระดับสุขภาพของคุณจากข้อมูลการออกกำลังกาย
    
    ### วิธีการใช้งาน
    1. กรอกข้อมูลในฟอร์มด้านขวา
    2. กดปุ่ม "ทำนายผล"
    3. ดูผลลัพธ์และคำแนะนำ
    
    ### ข้อมูลที่ใช้
    - ✅ ข้อมูลจริงจากแบบสำรวจ
    - ✅ โมเดล SVM ที่เทรนแล้ว
    - ✅ ความแม่นยำ ~60-70%
    """)
    
    st.markdown("---")
    st.markdown("**Developed by:** AI Expert Team")
    st.markdown("**Model:** SVM with RBF Kernel")

# ============================================================
# Main Content
# ============================================================
st.title("🔮 ทำนายระดับสุขภาพของคุณ")
st.markdown("กรอกข้อมูลการออกกำลังกายและไลฟ์สไตล์ของคุณ เพื่อรับคำทำนายระดับสุขภาพ")

# ============================================================
# สร้างฟอร์มกรอกข้อมูล
# ============================================================
st.markdown("## 📝 กรอกข้อมูลของคุณ")

# แบ่งเป็น 2 คอลัมน์
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 ข้อมูลพื้นฐาน")
    
    # อายุ
    age = st.slider("อายุ (ปี)", 15, 65, 25)
    
    # เพศ
    gender = st.selectbox(
        "เพศ",
        ["Male", "Female", "Other"],
        index=1
    )
    
    # ความสำคัญของการออกกำลังกาย
    importance = st.slider(
        "การออกกำลังกายสำคัญกับคุณแค่ไหน? (1-5)",
        1, 5, 3,
        help="1 = ไม่สำคัญเลย, 5 = สำคัญมาก"
    )
    
    # ระดับความฟิตปัจจุบัน
    fitness_level = st.selectbox(
        "ระดับความฟิตปัจจุบัน",
        ["Unfit", "Average", "Good", "Very good", "Perfect"],
        index=1
    )
    
    # ความถี่ในการออกกำลังกาย
    frequency = st.selectbox(
        "คุณออกกำลังกายบ่อยแค่ไหน?",
        ["Never", "1 to 2 times a week", "2 to 3 times a week", 
         "3 to 4 times a week", "5 to 6 times a week", "Everyday"],
        index=2
    )
    
    # ระยะเวลาออกกำลังกาย
    duration = st.selectbox(
        "คุณออกกำลังกายต่อวันนานแค่ไหน?",
        ["I don't really exercise", "30 minutes", "1 hour", 
         "2 hours", "3 hours and above"],
        index=1
    )

with col2:
    st.markdown("### 🏃 รายละเอียดการออกกำลังกาย")
    
    # ออกกำลังกายกับใคร
    with_whom = st.selectbox(
        "คุณออกกำลังกายกับใคร?",
        ["Alone", "With a friend", "With a group", 
         "Within a class environment", "I don't really exercise"],
        index=0
    )
    
    # เวลาที่ออกกำลังกาย
    time_of_day = st.selectbox(
        "คุณชอบออกกำลังกายเวลาไหน?",
        ["Early morning", "Afternoon", "Evening", "I don't really exercise"],
        index=0
    )
    
    # อาหาร
    diet = st.selectbox(
        "คุณกินอาหารที่ดีต่อสุขภาพไหม?",
        ["Yes", "Not always", "No"],
        index=1
    )
    
    # เคยแนะนำเพื่อนไหม
    recommended = st.selectbox(
        "คุณเคยแนะนำเพื่อนให้ออกกำลังกายไหม?",
        ["Yes", "No"],
        index=0
    )
    
    # เคยซื้ออุปกรณ์ไหม
    purchased = st.selectbox(
        "คุณเคยซื้ออุปกรณ์ออกกำลังกายไหม?",
        ["Yes", "No"],
        index=0
    )

# ============================================================
# Multi-select options (ใช้ checkboxes)
# ============================================================
st.markdown("---")
st.markdown("### 🎯 รายละเอียดเพิ่มเติม")

col3, col4 = st.columns(2)

with col3:
    st.markdown("**อุปสรรคในการออกกำลังกาย** (เลือกได้หลายข้อ)")
    barriers = st.multiselect(
        "เลือกอุปสรรค",
        ["I don't have enough time", "I can't stay motivated", 
         "I'll become too tired", "I don't really enjoy exercising",
         "I have an injury", "Cost", "Laziness"],
        default=[]
    )
    
    st.markdown("**รูปแบบการออกกำลังกาย** (เลือกได้หลายข้อ)")
    exercises = st.multiselect(
        "เลือกรูปแบบ",
        ["Walking or jogging", "Gym", "Swimming", "Yoga", 
         "Zumba dance", "Lifting weights", "Team sport"],
        default=[]
    )

with col4:
    st.markdown("**อุปสรรคด้านอาหาร** (เลือกได้หลายข้อ)")
    diet_barriers = st.multiselect(
        "เลือกอุปสรรค",
        ["Ease of access to fast food", "Temptation and cravings",
         "Lack of time", "Cost"],
        default=[]
    )
    
    st.markdown("**แรงจูงใจในการออกกำลังกาย** (เลือกได้หลายข้อ)")
    motivations = st.multiselect(
        "เลือกแรงจูงใจ",
        ["I want to be fit", "I want to lose weight",
         "I want to increase muscle mass and strength",
         "I want to be flexible", "I want to relieve stress",
         "I want to achieve a sporting goal"],
        default=[]
    )

# ============================================================
# ปุ่มทำนายผล
# ============================================================
st.markdown("---")
predict_button = st.button("🔮 ทำนายผล", type="primary", use_container_width=True)

# ============================================================
# ฟังก์ชันแปลงข้อมูล
# ============================================================
def prepare_input_data(age, gender, importance, fitness_level, frequency, 
                       duration, with_whom, time_of_day, diet, 
                       recommended, purchased, barriers, exercises, 
                       diet_barriers, motivations):
    """แปลงข้อมูลจากฟอร์มเป็น format ที่โมเดลใช้"""
    
    # แปลงเพศ
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    gender_encoded = gender_map[gender]
    
    # แปลงความถี่
    freq_map = {
        'Never': 0,
        '1 to 2 times a week': 1.5,
        '2 to 3 times a week': 2.5,
        '3 to 4 times a week': 3.5,
        '5 to 6 times a week': 5.5,
        'Everyday': 7
    }
    freq_numeric = freq_map[frequency]
    
    # แปลงระยะเวลา
    duration_map = {
        "I don't really exercise": 0,
        "30 minutes": 30,
        "1 hour": 60,
        "2 hours": 120,
        "3 hours and above": 180
    }
    duration_minutes = duration_map[duration]
    
    # แปลงกับใคร
    with_whom_map = {
        'Alone': 0,
        'With a friend': 1,
        'With a group': 2,
        'Within a class environment': 3,
        "I don't really exercise": -1
    }
    with_whom_encoded = with_whom_map[with_whom]
    
    # แปลงเวลา
    time_map = {
        'Early morning': 0,
        'Afternoon': 1,
        'Evening': 2,
        "I don't really exercise": -1
    }
    time_encoded = time_map[time_of_day]
    
    # แปลงอาหาร
    diet_map = {'Yes': 2, 'Not always': 1, 'No': 0}
    diet_encoded = diet_map[diet]
    
    # แปลงแนะนำเพื่อน
    recommended_encoded = 1 if recommended == 'Yes' else 0
    
    # แปลงซื้ออุปกรณ์
    purchased_encoded = 1 if purchased == 'Yes' else 0
    
    # แปลงระดับความฟิต
    fitness_map = {
        'Unfit': 1, 'Average': 2, 'Good': 3, 'Very good': 4, 'Perfect': 5
    }
    fitness_level_numeric = fitness_map[fitness_level]
    
    # สร้าง DataFrame
    data = {
        'age_numeric': age,
        'gender_encoded': gender_encoded,
        'freq_numeric': freq_numeric,
        'duration_minutes': duration_minutes,
        'with_whom_encoded': with_whom_encoded,
        'time_encoded': time_encoded,
        'diet_encoded': diet_encoded,
        'recommended_encoded': recommended_encoded,
        'purchased_encoded': purchased_encoded,
        'importance_numeric': importance,
        'fitness_level_numeric': fitness_level_numeric,
    }
    
    # เพิ่ม binary features จาก multi-select
    all_features = model_info['feature_columns']
    
    # เติม binary features ด้วย 0 ก่อน
    for col in all_features:
        if col not in data:
            data[col] = 0
    
    # ตรวจสอบว่าตัวเลือกอยู่ใน features หรือไม่
    for barrier in barriers:
        col_name = f"barrier_{barrier.replace(' ', '_')[:30]}"
        if col_name in all_features:
            data[col_name] = 1
    
    for exercise in exercises:
        col_name = f"exercise_{exercise.replace(' ', '_').replace('/', '_')[:30]}"
        if col_name in all_features:
            data[col_name] = 1
    
    for diet_barrier in diet_barriers:
        col_name = f"diet_barrier_{diet_barrier.replace(' ', '_')[:30]}"
        if col_name in all_features:
            data[col_name] = 1
    
    for motivation in motivations:
        col_name = f"motivation_{motivation.replace(' ', '_')[:30]}"
        if col_name in all_features:
            data[col_name] = 1
    
    # สร้าง DataFrame และเรียงคอลัมน์
    df = pd.DataFrame([data])
    df = df[all_features]
    
    return df

# ============================================================
# แสดงผลลัพธ์
# ============================================================
if predict_button:
    with st.spinner("กำลังทำนายผล..."):
        # แปลงข้อมูล
        input_data = prepare_input_data(
            age, gender, importance, fitness_level, frequency,
            duration, with_whom, time_of_day, diet,
            recommended, purchased, barriers, exercises,
            diet_barriers, motivations
        )
        
        # ทำนาย
        prediction = model.predict(input_data)[0]
       