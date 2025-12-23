"""
S-box Analysis Web Application (UPDATED WITH PHASE 4)
Streamlit app for S-box construction, validation, cryptographic testing, and AES encryption
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from PIL import Image

# Import custom modules
from sbox_core import SBoxConstructor, PredefinedMatrices
from sbox_validation import SBoxValidator
from sbox_testing import SBoxCryptoTest
from sbox_io import SBoxIO
from aes_cipher import AESCipher, AESImageCipher, generate_key_from_password

# Page configuration
st.set_page_config(
    page_title="Kahvi S-box Analysis & AES Encryption Tool",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sbox' not in st.session_state:
    st.session_state.sbox = None
if 'sbox_name' not in st.session_state:
    st.session_state.sbox_name = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'encrypted_text' not in st.session_state:
    st.session_state.encrypted_text = None
if 'encrypted_image' not in st.session_state:
    st.session_state.encrypted_image = None

def main():
    # Header
    st.markdown('<div class="main-header">🔐 S-box Analysis & AES Encryption Tool</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Based on:</b> "AES S-box modification uses affine matrices exploration for increased S-box strength"<br>
    <b>Features:</b> S-box Construction | Validation | Cryptographic Testing | AES Encryption/Decryption | Import/Export
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://unnes.ac.id/lppm/wp-content/uploads/sites/16/2015/08/Logo-Transparan-Warna-1.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Phase:",
            ["🔷 Phase 1: Construction",
             "🔷 Phase 2: Testing",
             "🔷 Phase 3: Import/Export",
             "🔷 Phase 4: Encryption"]  # NEW!
        )
        
        st.markdown("---")
        st.markdown("### Current S-box")
        if st.session_state.sbox is not None:
            st.success(f"✓ {st.session_state.sbox_name}")
            
            # Show S-box preview
            with st.expander("Preview (first row)"):
                st.code(str(st.session_state.sbox[0]))
            
            if st.button("🗑️ Clear S-box"):
                st.session_state.sbox = None
                st.session_state.sbox_name = None
                st.session_state.validation_results = None
                st.session_state.test_results = None
                st.rerun()
        else:
            st.info("No S-box loaded")
            st.markdown("💡 *Build one in Phase 1*")
    
    # Main content based on selected page
    if page == "🔷 Phase 1: Construction":
        show_phase1_construction()
    elif page == "🔷 Phase 2: Testing":
        show_phase2_testing()
    elif page == "🔷 Phase 3: Import/Export":
        show_phase3_io()
    elif page == "🔷 Phase 4: Encryption":
        show_phase4_encryption()  # NEW!

def show_phase1_construction():
    """Phase 1: S-box Construction"""
    st.markdown('<div class="sub-header">🔷 Phase 1: S-box Construction</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏗️ Build S-box", "📊 View Matrices", "✅ Validate"])
    
    with tab1:
        st.markdown("### 🏗️ S-box Construction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Affine matrix selection
            matrix_option = st.selectbox(
                "Select Affine Matrix:",
                ["AES (Original)", "K4 (Paper)", "K44 (Best - Paper)", 
                 "K81 (Paper)", "K111 (Paper)", "K128 (Paper)", "Custom"]
            )
            
            if matrix_option == "Custom":
                st.info("Enter custom 8×8 binary matrix (0s and 1s only)")
                custom_matrix_text = st.text_area(
                    "Custom Matrix (8 rows, 8 values per row):",
                    value="1 0 0 0 1 1 1 1\n" * 8,
                    height=200
                )
        
        with col2:
            st.markdown("#### Options")
            constant_option = st.selectbox(
                "8-bit Constant:",
                ["AES (0x63)", "Custom"]
            )
            
            if constant_option == "Custom":
                custom_constant = st.text_input(
                    "Enter 8-bit binary:",
                    value="11000110"
                )
        
        # Construct button
        if st.button("🔨 Construct S-box", type="primary", use_container_width=True):
            try:
                constructor = SBoxConstructor()
                
                # Get affine matrix
                if matrix_option == "AES (Original)":
                    affine_matrix = PredefinedMatrices.get_aes_matrix()
                    sbox_name = "AES S-box"
                elif matrix_option == "K4 (Paper)":
                    affine_matrix = PredefinedMatrices.get_k4()
                    sbox_name = "S-box K4"
                elif matrix_option == "K44 (Best - Paper)":
                    affine_matrix = PredefinedMatrices.get_k44()
                    sbox_name = "S-box K44"
                elif matrix_option == "K81 (Paper)":
                    affine_matrix = PredefinedMatrices.get_k81()
                    sbox_name = "S-box K81"
                elif matrix_option == "K111 (Paper)":
                    affine_matrix = PredefinedMatrices.get_k111()
                    sbox_name = "S-box K111"
                elif matrix_option == "K128 (Paper)":
                    affine_matrix = PredefinedMatrices.get_k128()
                    sbox_name = "S-box K128"
                else:  # Custom
                    # Parse custom matrix
                    rows = custom_matrix_text.strip().split('\n')
                    affine_matrix = np.array([[int(x) for x in row.split()] for row in rows], dtype=np.uint8)
                    sbox_name = "Custom S-box"
                
                # Get constant
                if constant_option == "AES (0x63)":
                    constant = PredefinedMatrices.get_aes_constant()
                else:
                    constant = np.array([[int(x) for x in custom_constant]], dtype=np.uint8).T
                
                # Construct S-box
                sbox = constructor.construct_sbox(affine_matrix, constant)
                
                # Store in session state
                st.session_state.sbox = sbox
                st.session_state.sbox_name = sbox_name
                
                st.success(f"✅ {sbox_name} constructed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Display current S-box
        if st.session_state.sbox is not None:
            st.markdown("---")
            st.markdown("### 📋 Current S-box")
            
            # Display options
            display_format = st.radio(
                "Display Format:",
                ["Decimal", "Hexadecimal", "Binary"],
                horizontal=True
            )
            
            # Convert to DataFrame for display
            df = pd.DataFrame(st.session_state.sbox)
            df.index = [f"Row {i}" for i in range(16)]
            df.columns = [f"Col {i}" for i in range(16)]
            
            if display_format == "Hexadecimal":
                df = df.applymap(lambda x: f"{x:02X}")
            elif display_format == "Binary":
                df = df.applymap(lambda x: f"{x:08b}")
            
            st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 View Predefined Matrices")
        
        matrix_to_view = st.selectbox(
            "Select Matrix:",
            ["K_AES", "K4", "K44", "K81", "K111", "K128"]
        )
        
        matrices = {
            "K_AES": PredefinedMatrices.get_aes_matrix(),
            "K4": PredefinedMatrices.get_k4(),
            "K44": PredefinedMatrices.get_k44(),
            "K81": PredefinedMatrices.get_k81(),
            "K111": PredefinedMatrices.get_k111(),
            "K128": PredefinedMatrices.get_k128()
        }
        
        matrix = matrices[matrix_to_view]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Matrix View")
            df = pd.DataFrame(matrix)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.markdown("#### Matrix Properties")
            st.write(f"**Determinant (mod 2):** {int(np.linalg.det(matrix)) % 2}")
            st.write(f"**Rank:** {np.linalg.matrix_rank(matrix)}")
            st.write(f"**Trace:** {np.trace(matrix)}")
    
    with tab3:
        st.markdown("### ✅ Validate S-box")
        
        if st.session_state.sbox is None:
            st.warning("⚠️ No S-box loaded. Please construct one first.")
            return
        
        if st.button("🔍 Run Validation Tests", type="primary"):
            with st.spinner("Running validation tests..."):
                validator = SBoxValidator()
                is_valid, results = validator.validate_sbox(st.session_state.sbox)
                st.session_state.validation_results = results
            
            if is_valid:
                st.success("✅ S-box is VALID (passes balance and bijective criteria)")
            else:
                st.error("❌ S-box is INVALID")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Balance Test")
                if results['balanced']:
                    st.success("✅ PASS")
                else:
                    st.error("✗ FAIL")
                
                # Balance details
                balance_df = pd.DataFrame([
                    {
                        'Bit': f'f{i}',
                        'Zeros': results['balance_details'][f'f{i}']['zeros'],
                        'Ones': results['balance_details'][f'f{i}']['ones'],
                        'Balanced': '✓' if results['balance_details'][f'f{i}']['balanced'] else '✗'
                    }
                    for i in range(8)
                ])
                st.dataframe(balance_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Bijective Test")
                if results['bijective']:
                    st.success("✅ PASS")
                else:
                    st.error("✗ FAIL")
                
                bij_details = results['bijective_details']
                st.write(f"**Unique values:** {bij_details['unique_count']}/256")
                st.write(f"**In range [0,255]:** {'Yes' if bij_details['in_range'] else 'No'}")

def show_phase2_testing():
    """Phase 2: Cryptographic Testing"""
    st.markdown('<div class="sub-header">🔷 Phase 2: Cryptographic Strength Testing</div>', unsafe_allow_html=True)
    
    if st.session_state.sbox is None:
        st.warning("⚠️ No S-box loaded. Please construct one in Phase 1.")
        return
    
    st.info(f"Testing: **{st.session_state.sbox_name}**")
    
    # Test options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_selection = st.multiselect(
            "Select Tests to Run:",
            ["NL", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP", "DU", "AD", "TO", "CI"],
            default=["NL", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP"]
        )
    
    with col2:
        st.markdown("#### Quick Select")
        if st.button("Select All"):
            test_selection = ["NL", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP", "DU", "AD", "TO", "CI"]
        if st.button("Core Tests"):
            test_selection = ["NL", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP"]
    
    if st.button("🧪 Run Selected Tests", type="primary", use_container_width=True):
        with st.spinner("Running cryptographic tests... This may take a few minutes."):
            tester = SBoxCryptoTest(st.session_state.sbox)
            
            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            test_map = {
                'NL': tester.test_nonlinearity,
                'SAC': tester.test_sac,
                'BIC-NL': tester.test_bic_nl,
                'BIC-SAC': tester.test_bic_sac,
                'LAP': tester.test_lap,
                'DAP': tester.test_dap,
                'DU': tester.test_differential_uniformity,
                'AD': tester.test_algebraic_degree,
                'TO': tester.test_transparency_order,
                'CI': tester.test_confusion_index
            }
            
            for i, test_name in enumerate(test_selection):
                status_text.text(f"Running {test_name}...")
                results[test_name] = test_map[test_name]()
                progress_bar.progress((i + 1) / len(test_selection))
            
            st.session_state.test_results = results
            status_text.text("✅ All tests completed!")
        
        st.success("✅ Testing completed!")
        st.rerun()
    
    # Display results
    if st.session_state.test_results:
        st.markdown("---")
        st.markdown("### 📊 Test Results")
        
        # Summary table
        summary_data = []
        for test_name, (value, details) in st.session_state.test_results.items():
            summary_data.append({
                'Test': test_name,
                'Value': f"{value:.5f}" if isinstance(value, float) else str(value),
                'Ideal': details.get('ideal', 'N/A'),
                'Score': f"{details.get('score', 0):.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Detailed views
        st.markdown("### 📈 Detailed Results")
        
        selected_test = st.selectbox("Select test for details:", list(st.session_state.test_results.keys()))
        
        value, details = st.session_state.test_results[selected_test]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Value", f"{value:.5f}" if isinstance(value, float) else str(value))
            st.metric("Ideal", details.get('ideal', 'N/A'))
        
        with col2:
            st.metric("Score", f"{details.get('score', 0):.2f}%")
            st.metric("Deviation", f"{details.get('deviation_from_ideal', 0):.5f}" if 'deviation_from_ideal' in details else "N/A")
        
        # Matrix visualizations
        if 'matrix' in details:
            st.markdown("#### Matrix Visualization")
            matrix = np.array(details['matrix'])
            
            # Replace None with NaN for visualization
            if matrix.dtype == object:
                matrix_clean = np.zeros_like(matrix, dtype=float)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        if matrix[i, j] is None:
                            matrix_clean[i, j] = np.nan
                        else:
                            matrix_clean[i, j] = float(matrix[i, j])
                matrix = matrix_clean
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                colorscale='RdYlGn',
                text=matrix,
                texttemplate='%{text:.3f}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                title=f"{selected_test} Matrix",
                xaxis_title="Bit",
                yaxis_title="Bit",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

def show_phase3_io():
    """Phase 3: Import/Export"""
    st.markdown('<div class="sub-header">🔷 Phase 3: Import/Export</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📥 Export", "📤 Import"])
    
    with tab1:
        st.markdown("### 📥 Export S-box")
        
        if st.session_state.sbox is None:
            st.warning("⚠️ No S-box loaded. Please construct one first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format:",
                ["Excel (.xlsx)", "CSV (.csv)", "Text (.txt)"]
            )
        
        with col2:
            number_format = st.selectbox(
                "Number Format:",
                ["Decimal", "Binary", "Hexadecimal"]
            )
        
        if st.button("💾 Generate Export File", type="primary"):
            try:
                if export_format == "Excel (.xlsx)":
                    formats = [number_format.lower()]
                    file_data = SBoxIO.export_to_excel(
                        st.session_state.sbox,
                        f"{st.session_state.sbox_name}.xlsx",
                        include_formats=formats
                    )
                    file_name = f"{st.session_state.sbox_name.replace(' ', '_')}.xlsx"
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
                elif export_format == "CSV (.csv)":
                    file_data = SBoxIO.export_to_csv(st.session_state.sbox, number_format.lower()).encode()
                    file_name = f"{st.session_state.sbox_name.replace(' ', '_')}.csv"
                    mime_type = "text/csv"
                
                else:  # Text
                    file_data = SBoxIO.export_to_txt(st.session_state.sbox, number_format.lower()).encode()
                    file_name = f"{st.session_state.sbox_name.replace(' ', '_')}.txt"
                    mime_type = "text/plain"
                
                st.download_button(
                    label="⬇️ Download File",
                    data=file_data,
                    file_name=file_name,
                    mime=mime_type,
                    use_container_width=True
                )
                
                st.success(f"✅ Export file ready for download!")
                
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    with tab2:
        st.markdown("### 📤 Import S-box")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['xlsx', 'csv', 'txt'],
            help="Upload S-box file in Excel, CSV, or Text format"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            try:
                if file_type == 'xlsx':
                    file_content = uploaded_file.read()
                    success, sbox, message = SBoxIO.import_from_excel(file_content)
                
                elif file_type == 'csv':
                    file_content = uploaded_file.read().decode()
                    success, sbox, message = SBoxIO.import_from_csv(file_content)
                
                else:  # txt
                    file_content = uploaded_file.read().decode()
                    success, sbox, message = SBoxIO.import_from_txt(file_content)
                
                if success:
                    st.session_state.sbox = sbox
                    st.session_state.sbox_name = f"Imported: {uploaded_file.name}"
                    st.success(f"✅ S-box imported successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Import failed: {message}")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

def show_phase4_encryption():
    """Phase 4: AES Encryption/Decryption (NEW!)"""
    st.markdown('<div class="sub-header">🔷 Phase 4: AES Encryption/Decryption</div>', unsafe_allow_html=True)
    
    if st.session_state.sbox is None:
        st.warning("⚠️ No S-box loaded. Please construct one in Phase 1 first.")
        st.info("💡 The S-box will be used for AES SubBytes transformation.")
        return
    
    st.success(f"✅ Using S-box: **{st.session_state.sbox_name}**")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Encryption", "🖼️ Image Encryption (Soon)", "ℹ️ About AES"])
    
    with tab1:
        show_text_encryption()
    
    with tab2:
        print("Coming Soon")
    
    with tab3:
        show_aes_info()

def show_text_encryption():
    """Text encryption/decryption interface"""
    st.markdown("### 📝 Text Encryption/Decryption")
    
    st.markdown("""
    <div class="info-box">
    <b>How it works:</b> Your text will be encrypted using AES-128 with the custom S-box.
    The same password must be used for encryption and decryption.
    </div>
    """, unsafe_allow_html=True)
    
    # Encryption/Decryption mode
    mode_selection = st.radio(
        "Select Mode:",
        ["🔒 Encrypt", "🔓 Decrypt"],
        horizontal=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        password = st.text_input(
            "Password (for key generation):",
            type="password",
            help="16-byte key will be derived from this password using SHA-256"
        )
    
    with col2:
        aes_mode = st.selectbox(
            "AES Mode:",
            ["ECB", "CBC"],
            help="ECB: Electronic Codebook, CBC: Cipher Block Chaining"
        )
    
    if mode_selection == "🔒 Encrypt":
        st.markdown("#### Input Message")
        plaintext = st.text_area(
            "Enter text to encrypt:",
            height=150,
            placeholder="Type your secret message here..."
        )
        
        if st.button("🔒 Encrypt Message", type="primary", use_container_width=True):
            if not plaintext:
                st.warning("⚠️ Please enter some text to encrypt.")
            elif not password:
                st.warning("⚠️ Please enter a password.")
            else:
                try:
                    # Generate key
                    key = generate_key_from_password(password)
                    
                    # Create cipher
                    cipher = AESCipher(st.session_state.sbox, key, mode=aes_mode)
                    
                    # Encrypt
                    ciphertext = cipher.encrypt(plaintext.encode('utf-8'))
                    ciphertext_hex = ciphertext.hex()
                    
                    st.session_state.encrypted_text = ciphertext_hex
                    
                    st.success("✅ Encryption successful!")
                    
                    # Display results
                    st.markdown("#### Encrypted Result")
                    st.code(ciphertext_hex, language=None)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Size", f"{len(plaintext)} bytes")
                    with col2:
                        st.metric("Encrypted Size", f"{len(ciphertext)} bytes")
                    with col3:
                        st.metric("Hex Length", f"{len(ciphertext_hex)} chars")
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Encrypted (HEX)",
                        data=ciphertext_hex,
                        file_name="encrypted_message.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Encryption failed: {str(e)}")
    
    else:  # Decrypt mode
        st.markdown("#### Encrypted Message")
        ciphertext_hex = st.text_area(
            "Enter encrypted text (HEX format):",
            height=150,
            value=st.session_state.encrypted_text if st.session_state.encrypted_text else "",
            placeholder="Paste encrypted hex string here..."
        )
        
        if st.button("🔓 Decrypt Message", type="primary", use_container_width=True):
            if not ciphertext_hex:
                st.warning("⚠️ Please enter encrypted text to decrypt.")
            elif not password:
                st.warning("⚠️ Please enter the password used for encryption.")
            else:
                try:
                    # Convert hex to bytes
                    ciphertext = bytes.fromhex(ciphertext_hex.strip())
                    
                    # Generate key
                    key = generate_key_from_password(password)
                    
                    # Create cipher
                    cipher = AESCipher(st.session_state.sbox, key, mode=aes_mode)
                    
                    # Decrypt
                    plaintext_bytes = cipher.decrypt(ciphertext)
                    plaintext = plaintext_bytes.decode('utf-8')
                    
                    st.success("✅ Decryption successful!")
                    
                    # Display results
                    st.markdown("#### Decrypted Message")
                    st.text_area("Result:", value=plaintext, height=150, disabled=True)
                    
                    st.metric("Message Length", f"{len(plaintext)} characters")
                    
                except Exception as e:
                    st.error(f"❌ Decryption failed: {str(e)}")
                    st.info("💡 Make sure you're using the correct password and AES mode.")

def show_image_encryption():
    """Image encryption/decryption interface"""
    st.markdown("### 🖼️ Image Encryption/Decryption")
    
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Note:</b> Image encryption is experimental. Works best with small images (< 1MB).
    Large images may take significant time to process.
    </div>
    """, unsafe_allow_html=True)
    
    mode_selection = st.radio(
        "Select Mode:",
        ["🔒 Encrypt Image", "🔓 Decrypt Image"],
        horizontal=True,
        key="image_mode"
    )
    
    password = st.text_input(
        "Password:",
        type="password",
        key="image_password",
        help="Use the same password for encryption and decryption"
    )
    
    if mode_selection == "🔒 Encrypt Image":
        uploaded_file = st.file_uploader(
            "Choose an image to encrypt:",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key="encrypt_upload"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")
            
            if st.button("🔒 Encrypt Image", type="primary", use_container_width=True):
                if not password:
                    st.warning("⚠️ Please enter a password.")
                else:
                    try:
                        with st.spinner("Encrypting image... This may take a moment."):
                            # Convert image to array
                            img_array = np.array(image)
                            
                            # Generate key
                            key = generate_key_from_password(password)
                            
                            # Create cipher
                            image_cipher = AESImageCipher(st.session_state.sbox, key, mode='ECB')
                            
                            # Encrypt (returns bytes now)
                            encrypted_bytes, metadata = image_cipher.encrypt_image(img_array)
                            
                            # Store in session
                            st.session_state.encrypted_image = {
                                'bytes': encrypted_bytes,
                                'metadata': metadata
                            }
                        
                        st.success("✅ Image encrypted successfully!")
                        
                        with col2:
                            st.markdown("#### Encrypted Data")
                            st.metric("Encrypted Size", f"{len(encrypted_bytes)} bytes")
                            st.metric("Original Size", f"{metadata['data_length']} bytes")
                            
                            # Create visual representation (random-looking image)
                            enc_array = np.frombuffer(encrypted_bytes[:metadata['data_length']], dtype=np.uint8)
                            try:
                                enc_img = enc_array.reshape(metadata['original_shape'])
                                encrypted_img = Image.fromarray(enc_img.astype('uint8'))
                                st.image(encrypted_img, caption="Visual representation", use_column_width=True)
                            except:
                                st.info("Visual representation not available (encrypted data)")
                            
                            # Download encrypted data
                            st.download_button(
                                label="⬇️ Download Encrypted Data",
                                data=encrypted_bytes,
                                file_name="encrypted_image.bin",
                                mime="application/octet-stream"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Encryption failed: {str(e)}")
    
    else:  # Decrypt mode
        if st.session_state.encrypted_image is not None:
            st.info("ℹ️ Using encrypted image from current session")
            
            if st.button("🔓 Decrypt Image", type="primary", use_container_width=True):
                if not password:
                    st.warning("⚠️ Please enter the password used for encryption.")
                else:
                    try:
                        with st.spinner("Decrypting image..."):
                            # Get encrypted data
                            encrypted_bytes = st.session_state.encrypted_image['bytes']
                            metadata = st.session_state.encrypted_image['metadata']
                            
                            # Generate key
                            key = generate_key_from_password(password)
                            
                            # Create cipher
                            image_cipher = AESImageCipher(st.session_state.sbox, key, mode='ECB')
                            
                            # Decrypt
                            decrypted_array = image_cipher.decrypt_image(encrypted_bytes, metadata)
                        
                        st.success("✅ Image decrypted successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Encrypted (Visual)")
                            # Show visual representation
                            enc_array = np.frombuffer(encrypted_bytes[:metadata['data_length']], dtype=np.uint8)
                            try:
                                enc_img = enc_array.reshape(metadata['original_shape'])
                                encrypted_img = Image.fromarray(enc_img.astype('uint8'))
                                st.image(encrypted_img, use_column_width=True)
                            except:
                                st.info("Visual not available")
                        
                        with col2:
                            st.markdown("#### Decrypted Image")
                            decrypted_img = Image.fromarray(decrypted_array.astype('uint8'))
                            st.image(decrypted_img, use_column_width=True)
                            
                            # Download decrypted
                            buf = BytesIO()
                            decrypted_img.save(buf, format='PNG')
                            st.download_button(
                                label="⬇️ Download Decrypted Image",
                                data=buf.getvalue(),
                                file_name="decrypted_image.png",
                                mime="image/png"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Decryption failed: {str(e)}")
                        st.info("💡 Make sure you're using the correct password.")
        else:
            st.info("ℹ️ No encrypted image in session. Please encrypt an image first.")

def show_aes_info():
    """Display information about AES"""
    st.markdown("### ℹ️ About AES with Custom S-box")
    
    st.markdown("""
    ## Advanced Encryption Standard (AES)
    
    AES is a symmetric block cipher that encrypts data in 128-bit blocks using keys of 128, 192, or 256 bits.
    This implementation uses **AES-128** with a **custom S-box**.
    
    ### 🔧 AES Structure
    
    AES consists of several rounds of transformations:
    
    1. **SubBytes** - Non-linear substitution using S-box (🔥 CUSTOM!)
    2. **ShiftRows** - Circular shift of rows
    3. **MixColumns** - Linear mixing of columns
    4. **AddRoundKey** - XOR with round key
    
    ### 🎯 Custom S-box Integration
    
    The **SubBytes** transformation uses your custom S-box instead of the standard AES S-box.
    This means:
    
    - ✅ Different diffusion properties
    - ✅ Potentially stronger against specific attacks
    - ✅ Unique encryption based on S-box choice
    
    ### 📊 Encryption Modes
    
    **ECB (Electronic Codebook)**
    - Each block encrypted independently
    - Fast and parallelizable
    - ⚠️ Same plaintext → same ciphertext
    
    **CBC (Cipher Block Chaining)**
    - Each block XORed with previous ciphertext
    - Sequential processing
    - ✅ Same plaintext → different ciphertext
    
    ### 🔐 Security Recommendations
    
    1. **Use strong passwords** (at least 12 characters)
    2. **Use CBC mode** for better security
    3. **Keep your S-box secret** if used in production
    4. **Test S-box strength** in Phase 2 before use
    
    ### 📚 Implementation Details
    
    - **Algorithm**: AES-128
    - **Block Size**: 128 bits (16 bytes)
    - **Key Size**: 128 bits (derived from password)
    - **Rounds**: 10
    - **Padding**: PKCS7
    - **S-box**: Custom (from Phase 1)
    
    ### ⚡ Performance Notes
    
    - Text encryption: Very fast
    - Image encryption: Slower (depends on size)
    - Recommended max image size: 1MB
    """)
    
    # Show current S-box info
    st.markdown("---")
    st.markdown("### 📋 Current S-box Info")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("S-box Name", st.session_state.sbox_name)
    
    with col2:
        if st.session_state.test_results and 'NL' in st.session_state.test_results:
            nl_value = st.session_state.test_results['NL'][0]
            st.metric("Nonlinearity", nl_value)
        else:
            st.metric("Nonlinearity", "Not tested")
    
    with col3:
        if st.session_state.test_results and 'BIC_SAC' in st.session_state.test_results:
            bic_sac = st.session_state.test_results['BIC_SAC'][0]
            st.metric("BIC-SAC", f"{bic_sac:.5f}")
        else:
            st.metric("BIC-SAC", "Not tested")

if __name__ == "__main__":
    main()