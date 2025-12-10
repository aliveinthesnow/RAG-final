import os
import streamlit as st
import streamlit.components.v1 as components

# --- BRIDGE SECRETS TO ENV ---
try:
    # On Streamlit Cloud, keys are in st.secrets. We move them to os.environ
    # so that brain.py (which uses os.getenv) can find them.
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    if "PINECONE_INDEX_NAME" in st.secrets:
        os.environ["PINECONE_INDEX_NAME"] = st.secrets["PINECONE_INDEX_NAME"]
except FileNotFoundError:
    pass # Running locally, uses .env file instead

from brain import DigitalSeniorBrain

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Last Brain Cell",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- HEADER & WARNING ---
st.markdown('<div class="app-title">Last Brain Cell</div>', unsafe_allow_html=True)

if "show_warning" not in st.session_state:
    st.session_state.show_warning = True

if st.session_state.show_warning:
    with st.container():
        st.warning("Last-Brain-Cell is still a beta app currently under development.\n\nSome answers may not be available or accurate at times.\n\nWe would highly appreciate your help in building this app. Please use the feedback buttons in the sidebar to help us out!", icon="🚧")
        if st.button("Dismiss Warning"):
            st.session_state.show_warning = False
            st.rerun()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* Hide Streamlit Header/Footer/Menu options if needed */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* header {visibility: hidden;} - Do not hide, or sidebar toggle disappears */
        
        /* Hide specifically the Deploy Button, Decoration, Main Menu (3 dots), and Status Widget */
        .stDeployButton {display: none !important;}
        .stAppDeployButton {display: none !important;}
        [data-testid="stDeployButton"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        
        /* Hide the 3-dots menu and status widget, but KEEP stToolbar visible for Sidebar Toggle */
        [data-testid="stMainMenu"] {visibility: hidden !important;}
        [data-testid="stStatusWidget"] {visibility: hidden !important;}
        
        /* [data-testid="stToolbar"] {visibility: hidden !important;}  <-- Removed to restore Sidebar Toggle */
        [data-testid="stHeaderActionElements"] {display: none !important;}
        
        /* Remove default padding - Adjusted to clear header */
        .block-container {
            padding-top: 6rem; /* Increased to ensure Title is visible below header */
            padding-bottom: 5rem;
        }
        
        /* Chat Bubble Styles */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 80px; /* Space for fixed input */
        }
        
        .user-bubble {
            align-self: flex-end;
            background-color: #2b313e; /* Dark Blue-Grey */
            color: #e3e3e3;
            padding: 10px 15px;
            border-radius: 15px 15px 0px 15px; /* Curved with one sharp corner */
            max-width: 70%;
            text-align: right;
            border: 1px solid #3c4043;
        }
        
        .bot-bubble {
            align-self: flex-start;
            background-color: #1e1f20; /* Dark Grey */
            color: #e3e3e3;
            padding: 10px 15px;
            border-radius: 15px 15px 15px 0px; /* Curved with one sharp corner */
            max-width: 70%;
            text-align: left;
            border: 1px solid #3c4043;
        }
        
        /* FORCE SIDEBAR ON TOP OF EVERYTHING */
        [data-testid="stSidebar"] {
            z-index: 9999999 !important;
        }

        /* Title Style - Moved to Top Bar */
        .app-title {
            position: fixed;
            top: 12px;
            left: 70px; /* Right of sidebar toggle */
            font-size: 24px;
            font-weight: bold;
            color: #e3e3e3;
            z-index: 999999; /* Higher than Header, Lower than Sidebar (forced above) */
        }
        
        /* Input Box Styling - Fixed at Bottom */
        .stTextInput {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background-color: #131314;
            z-index: 1000;
        }
        
        .stTextInput input {
            border-radius: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# --- INITIALIZE BRAIN ---
@st.cache_resource
def load_brain():
    return DigitalSeniorBrain()

try:
    brain = load_brain()
except Exception as e:
    st.error(f"Failed to load Brain: {e}")
    st.stop()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY CHAT HISTORY (Custom HTML) ---
chat_html = '<div class="chat-container">'
for message in st.session_state.messages:
    if message["role"] == "user":
        chat_html += f'<div class="user-bubble">{message["content"]}</div>'
    else:
        chat_html += f'<div class="bot-bubble">{message["content"]}</div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# --- CHAT INPUT ---
if prompt := st.chat_input("Type a message..."):
    # 1. Add User Message to State
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Force Rerun 
    st.rerun()

# --- GENERATE RESPONSE (After Rerun) ---
# Check if the last message is from user, if so, generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("..."):
        try:
            response = brain.generate_response(st.session_state.messages[-1]["content"])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- SIDEBAR FEEDBACK UI ---
with st.sidebar:
    st.markdown("### Help us improve!")
    st.link_button("Feedback / Suggestions", "https://docs.google.com/forms/d/e/1FAIpQLSfxdzEw_zo9EVwsA4KhPBDec8DD4vTHDMjD9WlN_zODyJSaCw/viewform?usp=dialog", use_container_width=True)
    st.link_button("Add Missing Information", "https://docs.google.com/forms/d/e/1FAIpQLSdNswehs6_ly46ZqutFTyzl45mBl7QPgu8yPtgPDT_SL62jaw/viewform?usp=dialog", use_container_width=True)
