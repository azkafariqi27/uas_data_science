import streamlit as st
import pandas as pd
import joblib

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Student Graduation Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

model_dt = joblib.load("model_uas_dt.pkl")
model_rf = joblib.load("model_uas_rf.pkl")
model_kmeans = joblib.load("model_kmeans.pkl")
scaler = joblib.load("scaler.pkl")

# =====================================================
# HEADER
# =====================================================

st.title("🎓 Student Graduation Prediction System")

st.markdown("""
### Sistem Prediksi Kelulusan Mahasiswa

Model yang digunakan:

- 🌳 Decision Tree
- 🌲 Random Forest
- 📊 K-Means Clustering
""")

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📚 Informasi")

    st.write("""
    Sistem ini digunakan untuk:

    ✅ Prediksi Kelulusan Mahasiswa

    ✅ Analisis Decision Tree

    ✅ Analisis Random Forest

    ✅ Pengelompokan Mahasiswa menggunakan K-Means
    """)

    st.divider()

    st.subheader("👨‍🎓 Data Science Project")

    st.write("""
    Machine Learning:
    - Decision Tree
    - Random Forest

    Clustering:
    - K-Means
    """)

# =====================================================
# FORM INPUT
# =====================================================

st.subheader("📋 Input Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    previous_grade = st.number_input(
        "Previous Grade",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    extracurricular = st.number_input(
        "Extracurricular Activities",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    parental_support = st.selectbox(
        "Parental Support",
        ["Low", "Medium", "High"]
    )

with col2:

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    online_classes = st.checkbox(
        "Mengikuti Kelas Online"
    )

# =====================================================
# ENCODING
# =====================================================

gender_val = 0 if gender == "Male" else 1

parental_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

parental_val = parental_map[parental_support]

online_classes_val = 1 if online_classes else 0

# =====================================================
# PREDIKSI
# =====================================================

if st.button("🔍 Prediksi Sekarang"):

    input_data = pd.DataFrame([[
        gender_val,
        previous_grade,
        extracurricular,
        parental_val,
        study_hours,
        attendance,
        online_classes_val
    ]],
    columns=[
        'Gender',
        'PreviousGrade',
        'ExtracurricularActivities',
        'ParentalSupport',
        'Study Hours',
        'Attendance (%)',
        'Online Classes Taken'
    ])

    # Normalisasi
    input_scaled = scaler.transform(input_data)

    # Prediksi
    pred_dt = model_dt.predict(input_scaled)[0]
    pred_rf = model_rf.predict(input_scaled)[0]
    pred_cluster = model_kmeans.predict(input_scaled)[0]

    st.divider()

    st.header("📊 Hasil Analisis")

    col_a, col_b = st.columns(2)

    # =================================================
    # DECISION TREE
    # =================================================

    with col_a:

        st.subheader("🌳 Decision Tree")

        if pred_dt == 1:
            st.success("LULUS")
        else:
            st.error("TIDAK LULUS")

    # =================================================
    # RANDOM FOREST
    # =================================================

    with col_b:

        st.subheader("🌲 Random Forest")

        if pred_rf == 1:
            st.success("LULUS")
        else:
            st.error("TIDAK LULUS")

    st.divider()

    # =================================================
    # K-MEANS
    # =================================================

    st.subheader("📊 Hasil Clustering")

    st.info(f"Mahasiswa termasuk dalam Cluster {pred_cluster}")

    if pred_cluster == 0:

        st.success("""
        Cluster 0

        Karakteristik:
        - Performa akademik cukup baik
        - Kehadiran stabil
        - Aktivitas belajar sedang
        """)

    elif pred_cluster == 1:

        st.success("""
        Cluster 1

        Karakteristik:
        - Nilai akademik tinggi
        - Dukungan keluarga baik
        - Potensi prestasi tinggi
        """)

    elif pred_cluster == 2:

        st.success("""
        Cluster 2

        Karakteristik:
        - Kehadiran tinggi
        - Aktif dalam pembelajaran daring
        - Keterlibatan belajar tinggi
        """)

    st.divider()

    # =================================================
    # KESIMPULAN
    # =================================================

    st.subheader("📌 Kesimpulan")

    if pred_rf == 1:

        st.success(
            "Mahasiswa diprediksi memiliki peluang tinggi untuk lulus."
        )

    else:

        st.warning(
            "Mahasiswa perlu meningkatkan performa akademiknya untuk meningkatkan peluang kelulusan."
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Data Science Project | Decision Tree • Random Forest • K-Means Clustering"
)
