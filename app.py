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
  im = Image.open("Logo Undip.png")
except FileNotFoundError:
  im = "📊"

st.set_page_config(
    page_title="Klasifikasi Area Deforestasi",
    page_icon=Image.open("Logo Undip.png") ,
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
    1. *Resize* ke 224x224
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
    st.session_state.uploaded_images = [] 
if 'active_files' not in st.session_state:
    st.session_state.active_files = []
if 'is_preprocessing_done' not in st.session_state:
    st.session_state.is_preprocessing_done = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'confirm_reset' not in st.session_state:
    st.session_state.confirm_reset = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

model = load_trained_model()

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("Logo Undip.png", use_container_width=True)
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

st.title("🌲 Klasifikasi Area Deforestasi")
st.markdown("Dikembangkan oleh **Agung Afrizal**")

with st.expander("ℹ️ Tentang Aplikasi dan Model"):
    st.markdown("""
    **Deskripsi Penelitian:**
    Aplikasi ini dirancang untuk melakukan klasifikasi area deforestasi berdasarkan citra satelit ke dalam dua kategori, yaitu deforestasi dan non-deforestasi. Tujuan utama aplikasi ini adalah membantu proses pemantauan dan identifikasi perubahan tutupan hutan secara otomatis dengan memanfaatkan pendekatan *deep learning*. Aplikasi ini dapat menjadi tolak ukur *Early Warning* aktivitas deforestasi sehingga dapat dilakukan mitigasi lebih awal.
    
    **Mekanisme Model:** 
    Model yang digunakan dalam aplikasi ini adalah ***Convolutional Neural Network*** **(CNN) dengan arsitektur MobileNetV2**, yang dikenal efisien secara komputasi dan cocok untuk pemrosesan citra berukuran besar seperti citra satelit. Untuk meningkatkan stabilitas konvergensi dan performa model, proses pelatihan dipadukan dengan algoritma optimasi **AMSGrad**, yaitu varian dari Adam yang menggunakan estimasi momen kedua yang lebih konservatif sehingga mampu mengurangi fluktuasi *learning rate* dan menjaga kestabilan proses pelatihan.

    Sistem mengklasifikasikan citra ke dalam dua kategori biner:
    * **Non-Deforestasi:** Meliputi area hutan (*Forest*), vegetasi herba, sungai, dan danau.
    * **Deforestasi:** Meliputi area lahan pertanian (*Annual/Permanent Crop*), padang rumput (*Pasture*), jalan raya, pemukiman, dan kawasan industri.

    **Akurasi Model Pengujian:**
    Dataset yang digunakan terdiri dari 134 citra satelit Google Earth wilayah Amerika Selatan, yang dibagi menjadi data *training* (51%), data *validation* (19%), dan *data testing* (30%).

    Model CNN MobileNetV2 yang telah dioptimasi dengan AMSGrad menunjukkan performa yang sangat baik. Model menghasilkan akurasi pengujian sebesar 100% dengan nilai *Loss Testing* 0,0108
    """)

    # BAGIAN KUNING (Ketentuan Penggunaan Data)
    st.warning("""
    **Ketentuan Penggunaan Data**
    
    1.  **Data** ***Upload*** **:** Pengguna dapat mengunggah data citra satelit hutan dengan format citra **.jpg**, **.jpeg**, **.png**, maupun **file ZIP (maksimal 100MB)** yang berisi *batch dataset*.
    2.  **Jenis Data:** Model dilatih menggunakan data citra satelit optik (*RGB*). Penggunaan foto objek non-geospasial (misal: foto benda atau manusia) akan menghasilkan prediksi yang tidak valid.
    3.  ***Pre-processing*** **:** Sistem akan secara otomatis melakukan *resizing* citra ke ukuran **224x224 piksel**, normalisasi nilai piksel (*rescaling* 1./255) sesuai standar *input layer* MobileNetV2, serta melakukan Augmentasi Data sebelum melakukan prediksi.
    """)

tab1, tab2, tab3 = st.tabs(["**📂 1.** ***Upload*** **Data**", "**🔍 2. Proses Klasifikasi**", "**📊 3. Laporan & Informasi**"])

with tab1:
    st.header("*Upload* Data Citra")
    st.write("Fitur ini mendukung *upload* **Satu Gambar**, **Banyak Gambar**, atau **File ZIP**.")

    # 1. Update parameter 'type' agar menerima ZIP
    uploaded_files = st.file_uploader(
        "Tarik file ke sini atau klik *browse*",
        type=["jpg", "png", "jpeg", "zip"], 
        accept_multiple_files=True,
        key=str(st.session_state.get('uploader_key', 0))
    )

    if uploaded_files:
        valid_images = []
        
        for file in uploaded_files:
            # Cek apakah file adalah ZIP
            if file.name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(file) as z:
                        for filename in z.namelist():
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                with z.open(filename) as f:
                                    # Baca file gambar dari ZIP ke memory
                                    img_bytes = io.BytesIO(f.read())
                                    img_bytes.name = filename
                                    valid_images.append(img_bytes)
                except Exception as e:
                    st.error(f"Gagal mengekstrak {file.name}: {e}")
            
            # Jika bukan ZIP (file gambar biasa)
            else:
                valid_images.append(file)
        
        # Simpan hasil akhir (list gambar) ke session state
        st.session_state.uploaded_images = valid_images
        
        if len(valid_images) > 0:
            st.success(f"✅ Berhasil memuat {len(valid_images)} citra.")

            # Preview gambar
            st.subheader("*Preview* Citra")
            cols = st.columns(5)
            for i, file in enumerate(valid_images[:10]):
                try:
                    img = Image.open(file)
                    with cols[i % 5]:
                        st.image(img, caption=file.name, use_container_width=True)
                except:
                    pass

            if len(valid_images) > 10:
                st.info(f"... dan {len(valid_images)-10} citra lainnya.")
        else:
            st.warning("⚠️ File ZIP kosong atau tidak berisi gambar yang didukung.")

        if not st.session_state.active_files and st.session_state.uploaded_images:
            st.session_state.active_files = st.session_state.uploaded_images.copy()
            st.session_state.is_preprocessing_done = False # Reset status pre-processing
          
with tab2:
    st.header("⚙️ *Pre-processing* & Klasifikasi")

    if not st.session_state.active_files:
        st.warning("⚠️ Belum ada data citra. Silakan upload di Tab 1. *Upload Data*")
    else:
        num_files = len(st.session_state.active_files)
        st.write(f"Terdapat **{num_files} data citra** dalam antrean.")

        if not st.session_state.is_preprocessing_done:
            st.info("Klik tombol di bawah untuk memulai tahapan *Pre-processing* (*Resizing* & Normalisasi).")

            if st.button("▶️ Lakukan Tahap *Pre-processing*", type="primary"):
                import time
                progress_text = "Melakukan *Resizing* ke 224x224 px..."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                
                time.sleep(0.5)
                my_bar.empty()
                
                st.session_state.is_preprocessing_done = True
                st.rerun()

        # --- VISUALISASI PIPELINE & VALIDASI DATA ---
        else:
            c_info, c_reset = st.columns([3, 1])
            with c_info:
                st.success("✅ *Pre-processing* Selesai. Silakan validasi data di bawah.")
            with c_reset:
                if st.button("🔄 Ulangi *Pre-processing*"):
                    st.session_state.is_preprocessing_done = False
                    st.rerun()

            st.divider()

            # INFO PIPELINE  ---
            st.subheader("1. **Pipeline *Pre-processing* (Inference)**")
            st.info("""
            **Catatan:**Augmentasi Data (Rotasi/*Flip*) hanya dilakukan saat fase ***Training*** **Data** dalam tahap pemodelan. 
            Pada fase aplikasi ini, citra diproses **tanpa distorsi** untuk menjaga keaslian data.
            """)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.success("✅ ***Resizing*** **(224x224)**\n\nMenyesuaikan dimensi input CNN MobileNetV2.")
            with c2:
                st.success("✅ **Normalisasi (1./255)**\n\nMengubah range piksel dari 0-255 ke 0-1.")
            with c3:
                st.success("✅ **Tensor Conversion**\n\nMengubah citra menjadi array 3D RGB (**Red**, **Green**, **Blue**) sesuai standar CNN MobileNetV2.")

            st.divider()

            # --- BAGIAN 2: PREVIEW DATA (BEFORE vs AFTER) ---
            st.subheader(f"2. Visualisasi Sebelum - Sesudah ({len(st.session_state.active_files)} Citra)")
            st.caption("Jika terdapat citra yang tidak sesuai (misal: gelap/rusak), klik tombol **Hapus** agar tidak diikutsertakan dalam klasifikasi.")

            for idx, file in enumerate(st.session_state.active_files):
                try:
                    file.seek(0)
                    img_original = Image.open(file)
                    img_resized = ImageOps.fit(img_original, (224, 224), Image.LANCZOS)
                    
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([2, 2, 1])
                        
                        with c1:
                            st.image(img_original, caption=f"Sebelum: Asli ({img_original.size[0]}x{img_original.size[1]})", use_container_width=True)
                        with c2:
                            st.image(img_resized, caption="Sesudah: Input (224x224)", width=150)
                        with c3:
                            st.write(f"**{file.name}**")
                            if st.button("🗑️ Hapus", key=f"del_{idx}_{file.name}", type="secondary"):
                                st.session_state.active_files.pop(idx)
                                st.rerun()
                                
                except Exception as e:
                    st.error(f"File rusak: {file.name}")

            st.divider()

            # --- BAGIAN 3: EKSEKUSI KLASIFIKASI ---
            if len(st.session_state.active_files) > 0:
                st.subheader("3. Klasifikasi Citra")
                st.write("Data sudah siap. Klik tombol di bawah untuk memulai pemindaian.")

                if st.button("🚀 Jalankan Prediksi Final", type="primary", use_container_width=True):
                    if model is None:
                        st.error("❌ Model gagal dimuat.")
                        st.stop()

                    results = []
                    progress_bar = st.progress(0)
                    total = len(st.session_state.active_files)

                    for idx, file in enumerate(st.session_state.active_files):
                        try:
                            file.seek(0)
                            img = Image.open(file)
                            processed_img = preprocess_image(img)
                            prediction = model.predict(processed_img, verbose=0)
                            prob_val = float(prediction[0][0])

                            if prob_val > 0.5:
                                label = "Deforestasi"; conf = prob_val
                            else:
                                label = "Non-Deforestasi"; conf = 1.0 - prob_val

                            results.append({
                                "Nama File": file.name,
                                "Prediksi": label,
                                "Tingkat Kepercayaan": conf,
                                "Probabilitas Deforestasi": prob_val
                            })

                        except Exception as e:
                            st.error(f"Error: {e}")
                        
                        progress_bar.progress((idx + 1) / total)

                    progress_bar.empty()

                    if len(results) > 0:
                        df_results = pd.DataFrame(results)
                        df_display = df_results.copy()
                        
                        if 'Tingkat Kepercayaan' in df_display.columns:
                            df_display['Tingkat Kepercayaan'] = df_display['Tingkat Kepercayaan'].apply(lambda x: f"{x*100:.2f}%")

                        st.session_state.prediction_results = df_results
                        st.session_state.display_results = df_display
                        
                        st.success("✅ Analisis Selesai! Silakan buka Tab 3. Laporan & Informasi untuk hasil detail.")
                        st.balloons()
            else:
                st.warning("⚠️ Semua data telah dihapus. Silakan upload ulang di Tab 1. *Upload Data*")

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
                label="📥 Unduh CSV",
                data=csv,
                file_name='hasil_prediksi_deforestasi.csv',
                mime='text/csv',
                use_container_width=True
            )

            excel_data = convert_df_to_excel(df_res)
            st.download_button(
                label="📥 Unduh Excel",
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
