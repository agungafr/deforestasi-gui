# Modul
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import io
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import zipfile

try:
  im = Image.open("Logo UNDIP.png")
except FileNotFoundError:
  im = "📊"

st.set_page_config(
    page_title="Klasifikasi Area Deforestasi",
    page_icon="Logo Undip.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 200px;
        max-width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_trained_model():
    # Memuat model CNN MobileNetV2
    try:
        model_path = 'model_mobilenetv2_after.keras'
        
        # DEBUG: Cek apakah file ada
        if not os.path.exists(model_path):
            st.error(f"❌ File model '{model_path}' **tidak ditemukan** di direktori ini!")
            st.write(f"Isi folder saat ini: {os.listdir('.')}")
            return None
            
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ File ditemukan tapi **gagal diproses**. Error: {e}")
        return None

def convert_df_to_csv(df):
    """Mengubah DataFrame menjadi file CSV."""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Mengubah DataFrame menjadi file Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
    processed_data = output.getvalue()
    return processed_data

def preprocess_image(image_data):
    """
    Preprocessing citra agar sesuai input MobileNetV2:
    1. Resize ke 224x224
    2. Konversi ke Array
    3. Normalisasi (1./255)
    """
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)

    # Pastikan RGB
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    normalized_img = img_array / 255.0
    batch_img = np.expand_dims(normalized_img, axis=0)
    return batch_img

### Inisialisasi Session State

if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = [] # Menyimpan objek file gambar
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None # Menyimpan hasil DataFrame
if 'confirm_reset' not in st.session_state:
    st.session_state.confirm_reset = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

model = load_trained_model()

with st.sidebar:
    st.image("Logo UNDIP.png", width=70)
    st.title("Panel Kontrol")
    st.info("""
    **Identitas Pengembang:**
    * **Nama:** Agung Afrizal
    * **NIM:** 24050121120028
    * **Prodi:** Statistika UNDIP

    **Spesifikasi Model CNN:**
    * **Arsitektur:** MobileNetV2
    * **Optimasi:** AMSGrad
    * **Input:** Citra Satelit
    """)


tab1, tab2, tab3 = st.tabs(["**📂 1. Upload Data**", "**🔍 2. Proses Klasifikasi**", "**📊 3. Laporan & Info**"])

with tab1:
    st.header("Upload Data Citra")
    st.write("Fitur ini mendukung upload **Satu Gambar (Tunggal)** maupun **Banyak Gambar (Batch)** sekaligus.")

    uploaded_files = st.file_uploader(
        "Tarik file ke sini atau klik browse",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key=str(st.session_state.uploader_key)
    )

    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.success(f"✅ {len(uploaded_files)} Citra berhasil dimuat.")

        # Preview gambar
        st.subheader("Preview Citra")
        cols = st.columns(5)
        for i, file in enumerate(uploaded_files[:10]):
            img = Image.open(file)
            with cols[i % 5]:
                st.image(img, caption=file.name, use_container_width=True)

        if len(uploaded_files) > 10:
            st.info(f"... dan {len(uploaded_files)-10} citra lainnya.")

with tab2:
    st.header("Hasil Klasifikasi")

    if not st.session_state.uploaded_images:
        st.warning("⚠️ Silakan upload gambar terlebih dahulu di Tab Upload Data.")
    else:
        st.write(f"Siap melakukan klasifikasi pada **{len(st.session_state.uploaded_images)}** data citra.")

        if st.button("🚀 Jalankan Prediksi", type="primary"):
            # CEK MODEL DULU: Mencegah error 'NoneType'
            if model is None:
                st.error("❌ Model gagal dimuat. Cek apakah file model sudah ada di GitHub dan ukurannya < 100MB.")
                st.stop()

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            total = len(st.session_state.uploaded_images)

            for idx, file in enumerate(st.session_state.uploaded_images):
                status_text.text(f"Memproses: {file.name} ({idx+1}/{total})")

                try:
                    # 1. Reset pointer & Baca Gambar
                    file.seek(0)
                    img = Image.open(file)
                    processed_img = preprocess_image(img)

                    # 2. Prediksi
                    prediction = model.predict(processed_img, verbose=0)
                    prob_val = float(prediction[0][0])

                    # 3. Tentukan Label
                    if prob_val > 0.5:
                        label = "Deforestasi"
                        conf = prob_val
                    else:
                        label = "Non-Deforestasi"
                        conf = 1.0 - prob_val

                    # 4. Simpan ke variabel
                    item_hasil = {
                        "Nama File": file.name,
                        "Prediksi": label,
                        "Confidence": conf,
                        "Probabilitas Raw": prob_val
                    }
                    
                    # 5. Masukkan ke list
                    results.append(item_hasil)

                except Exception as e:
                    st.error(f"Gagal memproses {file.name}. Error: {str(e)}")

                # Update progress
                progress_bar.progress((idx + 1) / total)

            # Selesai loop
            status_text.text("Klasifikasi Selesai!")
            progress_bar.empty()

            # Cek hasil
            if len(results) > 0:
                df_results = pd.DataFrame(results)
                df_display = df_results.copy()
                
                if 'Confidence' in df_display.columns:
                    df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x*100:.2f}%")

                st.session_state.prediction_results = df_results
                st.session_state.display_results = df_display

                st.success("✅ Selesai! Cek hasil detail di Tab 3. Laporan dan Info")
            else:
                st.warning("⚠️ Tidak ada gambar yang berhasil diproses.")

with tab3:
    st.header("Laporan Hasil & Statistik")

    if st.session_state.prediction_results is None:
        st.info("Data belum diproses. Lakukan klasifikasi di Tab 2. Proses Klasifikasi.")
    else:
        df_res = st.session_state.prediction_results
        df_show = st.session_state.display_results

        # 1. Metrik Ringkasan
        col1, col2, col3 = st.columns(3)
        n_defor = len(df_res[df_res['Prediksi'] == 'Deforestasi'])
        n_non = len(df_res[df_res['Prediksi'] == 'Non-Deforestasi'])

        col1.metric("Total Data", len(df_res))
        col2.metric("Deforestasi", n_defor, delta_color="inverse")
        col3.metric("Non-Deforestasi", n_non)

        st.divider()

        # 2. Visualisasi Grafik & Tabel
        c_chart, c_table = st.columns([1, 2])

        # --- BAGIAN GRAFIK (Fix Indentasi di sini) ---
        with c_chart:
            st.subheader("Grafik Sebaran")
            fig, ax = plt.subplots(figsize=(4,5))
            
            # Warna custom: Hijau & Merah
            colors = ['#B22B27', '#71BC68'] 
            bars = ax.bar(['Non-Defor', 'Defor'], [n_non, n_defor], color=colors)

            # Menambahkan angka di atas batang
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

            ax.set_title("Distribusi Kelas")
            ax.set_ylabel("Jumlah Citra")
            
            # Mempercantik grafik (hapus garis atas & kanan)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # PENTING: st.pyplot dipanggil SEKALI saja dan DI DALAM with c_chart
            st.pyplot(fig, use_container_width=True)

        # --- BAGIAN TABEL ---
        with c_table:
            st.subheader("Tabel Detail")
            st.dataframe(df_show, use_container_width=True, height=350)

            # Tombol Download
            csv = convert_df_to_csv(df_res)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name='hasil_prediksi_deforestasi.csv',
                mime='text/csv',
                use_container_width=True
            )

            excel_data = convert_df_to_excel(df_res)
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name='hasil_prediksi_deforestasi.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )

    # Tombol Reset
    st.divider()
    if st.button("🔄 Reset Aplikasi"):
        st.session_state.uploaded_images = []
        st.session_state.prediction_results = None
        st.session_state.display_results = None

        st.session_state.uploader_key += 1
        st.rerun()
