import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os
import io

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="MorphoX Lab - UAS Modul 5",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔬"
)

# ==================== CSS STYLING ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

html, body, [class*="css"] { 
    font-family: 'Poppins', sans-serif; 
    color: #E0E0E0; 
}

.stApp { 
    background: linear-gradient(-45deg, #0d0221, #150734, #1a0a2e, #240b36);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
}

@keyframes gradientBG {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

section[data-testid="stSidebar"] { display: none; }

.main-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ff00ff, #00ffff, #ff00ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    animation: shine 3s linear infinite;
    letter-spacing: 3px;
}

@keyframes shine { to { background-position: 200% center; } }

.sub-title {
    text-align: center;
    color: #b0b0b0;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #00ffff;
    text-align: center;
    margin: 20px 0;
    text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
}

.image-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(255, 0, 255, 0.2);
    margin-bottom: 10px;
}

.card-header {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.card-header-input { color: #00ffff; }
.card-header-output { color: #ff00ff; }

.filter-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff00ff, #00ffff);
    color: white;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(135deg, #ff00ff, #00ffff);
    color: white;
    border-radius: 25px;
    width: 100%;
    font-weight: 600;
    padding: 12px 25px;
    border: none;
    transition: all 0.3s ease;
}

div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(0, 255, 255, 0.4);
}

.info-box {
    background: rgba(0, 255, 255, 0.1);
    border-left: 3px solid #00ffff;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}

.welcome-box {
    background: rgba(255,0,255,0.05);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    border: 1px solid rgba(0, 255, 255, 0.2);
}

.feature-card {
    background: rgba(255,255,255,0.03);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid rgba(0, 255, 255, 0.2);
    text-align: center;
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255, 0, 255, 0.5);
}

.feature-card h3 { font-size: 2.5rem; margin-bottom: 10px; }
.feature-card h4 { color: #00ffff; font-weight: 600; }

.team-info {
    background: rgba(255, 0, 255, 0.1);
    border-radius: 10px;
    padding: 10px 20px;
    display: inline-block;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff00ff, #00ffff, transparent);
    margin: 30px 0;
}

[data-testid="stImage"] {
    border-radius: 15px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================== FUNGSI MORFOLOGI ====================
def process_image(img, operation, params=None):
    """Proses gambar dengan operasi morfologi"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if operation == 'Original':
        return img_rgb
    
    kernel_size = params.get('kernel', 5) if params else 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'Dilasi':
        iterations = params.get('iterations', 1) if params else 1
        return cv2.dilate(img_rgb, kernel, iterations=iterations)
    
    elif operation == 'Erosi':
        iterations = params.get('iterations', 1) if params else 1
        return cv2.erode(img_rgb, kernel, iterations=iterations)
    
    elif operation == 'Opening':
        return cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)
    
    elif operation == 'Closing':
        return cv2.morphologyEx(img_rgb, cv2.MORPH_CLOSE, kernel)
    
    elif operation == 'Region Filling':
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to binary using Otsu's method
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check if background is white (assuming top-left corner is background)
        # If background is white (255), invert the image so background becomes black (0)
        if binary[0, 0] == 255:
            binary = cv2.bitwise_not(binary)
        
        # Copy the binary image
        im_floodfill = binary.copy()
        
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels larger than the image.
        h, w = binary.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        
        # Combine the two images to get the foreground with filled holes
        im_out = binary | im_floodfill_inv
        
        # Konversi ke RGB
        result = cv2.cvtColor(im_out, cv2.COLOR_GRAY2RGB)
        
        return result
    
    return img_rgb

# ==================== FUNGSI PDF ====================
def create_pdf(original_rgb, processed, op_name):
    """Buat laporan PDF"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(138, 43, 226)
    pdf.cell(0, 15, txt="MorphoX Lab", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, txt="Laporan Morfologi Citra - Modul 5", ln=True, align='C')
    pdf.cell(0, 8, txt="Kelompok 8 - UAS", ln=True, align='C')
    
    pdf.ln(5)
    pdf.set_draw_color(138, 43, 226)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, txt=f"Operasi: {op_name}", ln=True, align='L')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig:
        Image.fromarray(original_rgb).save(tmp_orig.name)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(0, 191, 255)
        pdf.cell(0, 10, txt="Gambar Original:", ln=True)
        pdf.image(tmp_orig.name, x=30, w=150)
        path_orig = tmp_orig.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_proc:
        Image.fromarray(processed).save(tmp_proc.name)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(255, 0, 255)
        pdf.cell(0, 10, txt="Gambar Hasil:", ln=True)
        pdf.image(tmp_proc.name, x=30, w=150)
        path_proc = tmp_proc.name
    
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, txt="Generated by MorphoX Lab - UAS Modul 5", ln=True, align='C')

    output = pdf.output(dest='S').encode('latin-1')
    os.remove(path_orig)
    os.remove(path_proc)
    return output

# ==================== DATA OPERASI ====================
OPERATIONS = {
    'Original': {'icon': '🖼️', 'desc': 'Gambar asli'},
    'Dilasi': {'icon': '➕', 'desc': 'Memperbesar objek'},
    'Erosi': {'icon': '➖', 'desc': 'Memperkecil objek'},
    'Opening': {'icon': '🧹', 'desc': 'Hapus noise kecil'},
    'Closing': {'icon': '🩹', 'desc': 'Tutup lubang kecil'},
    'Region Filling': {'icon': '🫧', 'desc': 'Isi area dalam objek'}
}

# ==================== UI UTAMA ====================
st.markdown('<h1 class="main-title">🔬 MORPHOX LAB</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">UAS Modul 5 - Morfologi Citra Digital | Kelompok 8</p>', unsafe_allow_html=True)

# Upload Section
st.markdown('<p class="section-title">📤 UPLOAD GAMBAR</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility="collapsed")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    st.markdown("---")
    
    # Pilih Operasi
    st.markdown('<p class="section-title">🎨 PILIH OPERASI</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        operation = st.selectbox(
            "Operasi",
            list(OPERATIONS.keys()),
            format_func=lambda x: f"{OPERATIONS[x]['icon']} {x}",
            label_visibility="collapsed"
        )
        
        st.markdown(f"""
            <div class="info-box">
                <div style="font-size: 2rem; text-align: center;">{OPERATIONS[operation]['icon']}</div>
                <h3 style="color: #00ffff; text-align: center;">{operation}</h3>
                <p style="text-align: center; color: #888;">{OPERATIONS[operation]['desc']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Parameter
    params = {}
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if operation in ['Dilasi', 'Erosi', 'Opening', 'Closing']:
            st.markdown("##### 🔧 Ukuran Kernel")
            params['kernel'] = st.slider("Kernel", 3, 21, 5, step=2, label_visibility="collapsed")
        
        if operation in ['Dilasi', 'Erosi']:
            st.markdown("##### 🔄 Iterasi")
            params['iterations'] = st.slider("Iterasi", 1, 10, 1, label_visibility="collapsed")
        
        # Region Filling menggunakan Otsu's method (threshold otomatis)
    
    # Process
    processed = process_image(img_bgr, operation, params)
    
    st.markdown("---")
    
    # Hasil
    st.markdown('<p class="section-title">🖼️ HASIL</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="image-card"><div class="card-header card-header-input">📥 INPUT</div></div>', unsafe_allow_html=True)
        st.image(img_rgb_orig, width=400)
    
    with col2:
        st.markdown(f'<div class="image-card"><div class="card-header card-header-output">📤 OUTPUT</div><span class="filter-badge">{OPERATIONS[operation]["icon"]} {operation}</span></div>', unsafe_allow_html=True)
        st.image(processed, width=400)
    
    st.markdown("---")
    
    # Statistik
    st.markdown('<p class="section-title">📊 STATISTIK</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Width", f"{img_rgb_orig.shape[1]} px")
    c2.metric("Height", f"{img_rgb_orig.shape[0]} px")
    c3.metric("Channels", f"{img_rgb_orig.shape[2] if len(img_rgb_orig.shape) > 2 else 1}")
    c4.metric("Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")
    
    st.markdown("---")
    
    # Download
    st.markdown('<p class="section-title">💾 DOWNLOAD</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        buf = io.BytesIO()
        Image.fromarray(processed).save(buf, format='PNG')
        
        st.download_button(
            "🖼️ Download PNG",
            buf.getvalue(),
            f"MorphoX_{operation}.png",
            "image/png",
            use_container_width=True
        )
        
        if st.button("📄 Generate PDF", use_container_width=True):
            st.session_state['pdf_data'] = create_pdf(img_rgb_orig, processed, operation)
            st.session_state['pdf_ready'] = True
            st.success("✅ PDF siap!")
        
        if st.session_state.get('pdf_ready'):
            st.download_button(
                "📥 Download PDF",
                st.session_state['pdf_data'],
                f"MorphoX_{operation}.pdf",
                "application/pdf",
                use_container_width=True
            )

else:
    st.markdown("---")
    st.markdown("""
        <div class="welcome-box">
            <h2 style="color: #fff;">🖼️ Selamat Datang!</h2>
            <p style="color: #b0b0b0;">Upload gambar untuk memulai operasi morfologi</p>
            <p style="color: #888; font-size: 0.9rem;">
                ➕ Dilasi • ➖ Erosi • 🧹 Opening • 🩹 Closing • 🫧 Region Filling
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p class="section-title">✨ FITUR</p>', unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
            <div class="feature-card">
                <h3>➕ ➖</h3>
                <h4>Dilasi & Erosi</h4>
                <p style="color: #888;">Perbesar & Perkecil Objek</p>
            </div>
        """, unsafe_allow_html=True)
    
    with f2:
        st.markdown("""
            <div class="feature-card">
                <h3>🧹 🩹</h3>
                <h4>Opening & Closing</h4>
                <p style="color: #888;">Hapus Noise & Tutup Lubang</p>
            </div>
        """, unsafe_allow_html=True)
    
    with f3:
        st.markdown("""
            <div class="feature-card">
                <h3>🫧</h3>
                <h4>Region Filling</h4>
                <p style="color: #888;">Isi Area dalam Objek</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 15px;">
        <div class="team-info">
            <span style="color: #00ffff; font-weight: 600;">👥 Kelompok 8</span>
            <span style="color: #666;"> • </span>
            <span style="color: #ff00ff;">UAS Modul 5 - Morfologi Citra Digital</span>
        </div>
    </div>
""", unsafe_allow_html=True)
