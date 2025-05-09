import streamlit as st
from PIL import Image
import io
from ollama_utils import OllamaClient

# Page configuration
st.set_page_config(
    page_title="OCR Assistant - AUI Academic Project",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Ollama client
@st.cache_resource
def get_ollama_client():
    return OllamaClient()

client = get_ollama_client()

# Header
st.title("🔍 AUI OCR Assistant")
st.markdown("""
Welcome to the **Gemma OCR Assistant**, an academic project developed at **Al Akhawayn University**.

This tool uses **CRNN (Convolutional Recurrent Neural Network)** models for Optical Character Recognition (OCR),
combined with **Gemma 3B via Ollama** for intelligent feedback and grading automation.

**Key Features:**
- Automatically extracts text from uploaded images
- Supports custom prompts for contextual analysis
- Designed for educational use in grading handwritten or printed student submissions
""")

# Sidebar with instructions
with st.sidebar:
    st.header("📝 Instructions")
    st.markdown("""
    1. Upload an image containing text
    2. (Optional) Provide a custom prompt to guide AI analysis
    3. Click 'Analyze Image' to run CRNN + LLM OCR processing

    **Example Prompts:**
    - Extract and list all text from the image
    - Describe the layout and formatting of text
    - Analyze the context and meaning of the text
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Image Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            image = None
    
    custom_prompt = st.text_area(
        "🧠 Custom Prompt (optional)",
        placeholder="Enter a custom prompt to guide the analysis...",
        help="Leave empty to use the default prompt for general text extraction and analysis."
    )

with col2:
    st.subheader("📊 Analysis Results")
    if uploaded_file is not None and st.button("Analyze Image"):
        with st.spinner("🔄 Processing image with CRNN + Gemma..."):
            try:
                result = client.analyze_image(image, custom_prompt)
                st.markdown("### ✅ Results:")
                st.write(result)
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
🔬 *This application is part of an academic research project at* **Al Akhawayn University**, *utilizing* **CRNN models** *for text extraction and* **Gemma 3B** *for automated feedback.*

🔗 *Powered by, *Streamlit, and open-source vision-language AI.*
""")
