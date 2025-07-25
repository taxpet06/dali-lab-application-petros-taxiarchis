import streamlit as st

# Initialize
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Home"

# Define pages
import streamlit as st

def home():
    st.title("Home")
    st.write("Instructions")

def load():
    st.title("Load")
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    # Button
    if st.button("Load Image"):
        st.session_state.show_uploader = True  # flag to show uploading widget
    
    # Only show uploader if button was clicked
    if st.session_state.get('show_uploader', False):
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            key="image_uploader"
        )
        
        # Update session state if a file is uploaded
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file # set uploaded file
            st.session_state.show_uploader = False # hide uploader
            st.rerun() # Refresh
    
    # Display the image
    if st.session_state.uploaded_file is not None:
        st.image(
            st.session_state.uploaded_file, 
            caption=f"Uploaded: {st.session_state.uploaded_file.name}",
            use_container_width=True
        )
        st.success("Image loaded successfully.")
        
        # Clear image
        if st.button("Remove Image"):
            st.session_state.uploaded_file = None
            st.rerun()  # Refresh 

def metrics():
    st.title("Metrics")

def images():
    st.title("Images")

# Sidebar 
with st.sidebar:
    st.header("Navigation")
    if st.button("Home"):
        st.session_state.current_screen = "Home"
    if st.button("Load"):
        st.session_state.current_screen = "Load"
    if st.button("Metrics"):
        st.session_state.current_screen = "Metrics"
    if st.button("Images"):
        st.session_state.current_screen = "Images"

# Display current screen
if st.session_state.current_screen == "Home":
    home()
elif st.session_state.current_screen == "Metrics":
    metrics()
elif st.session_state.current_screen == "Images":
    images()
elif st.session_state.current_screen == "Load":
    load()