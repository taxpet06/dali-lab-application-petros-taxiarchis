import streamlit as st
from PIL import Image
import io


# Initialize
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Home"

# Define pages
import streamlit as st

def home():
    st.title("Home")
    st.markdown("""
        **Instructions**:
        
        1.  Navigate to the ***Train*** tab
        2.  Load your images and masks accordingly  
                - Both images and masks do not need to be loaded in proper order as long as their filename ends with a number  
                - You may use the ***Remove all images*** or ***Remove all masks*** buttons if a mistake is made
        3.  Use the ***Train*** button and save the trained model
        4.  Navigate to the ***Test*** tab
        5.  Load the either the model that was just trained or a model of your choice
        6.  Load testing data
        7.  Use the ***Segment*** button to perform barnacle classification
        8.  The *Metrics* tab will display a variety of useful information and graphs regarding your training and testing datasets
        9.  The ***Images*** tab will display the images and the overlayed masks for your training and testing datasets
        """
    )

def train():
    st.title("Train")
    
    #---Images---#
    # Initialization of list
    if 'training_images' not in st.session_state:
        st.session_state.training_images = []
    
    # Button to trigger file uploader
    if st.button("Load Image(s)"):
        st.session_state.show_uploader = True
    
    # File uploade
    if st.session_state.get('show_uploader', False):
        training_images = st.file_uploader(
            "Choose images", 
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            accept_multiple_files=True,
            key="image_uploader"
        )
        
        if training_images:
            st.session_state.training_images.extend(training_images)
            st.session_state.show_uploader = False
            st.rerun()
    
    # Display images in carousel if any exist
    if st.session_state.training_images:
        st.success(f"{len(st.session_state.training_images)} image(s) loaded")
        
        # Create a carousel with arrows to navigate
        cols = st.columns([1, 4, 1])
        with cols[1]:
            if 'current_image_index' not in st.session_state:
                st.session_state.current_image_index = 0
            
            current_img = st.session_state.training_images[st.session_state.current_image_index]
            st.image(
                current_img,
                caption=f"Image {st.session_state.current_image_index + 1}/{len(st.session_state.training_images)}: {current_img.name}",
                use_container_width=True
            )
        
        # Navigation arrows
        with cols[0]:
            if st.button("<-"):
                st.session_state.current_image_index = max(0, st.session_state.current_image_index - 1)
                st.rerun()
        
        with cols[2]:
            if st.button("->"):
                st.session_state.current_image_index = min(len(st.session_state.training_images) - 1, st.session_state.current_image_index + 1)
                st.rerun()
        
        # Clear all button
        if st.button("Remove All Images"):
            st.session_state.training_images = []
            if 'current_image_index' in st.session_state:
                del st.session_state.current_image_index
            st.rerun() # Refresh

    #---Masks---#        
    # Initialization of list
    if 'training_masks' not in st.session_state:
        st.session_state.training_masks = []
    
    # Button to trigger file uploader
    if st.button("Load Mask(s)"):
        st.session_state.show_uploader2 = True
    
    # File uploade
    if st.session_state.get('show_uploader2', False):
        training_masks = st.file_uploader(
            "Choose images", 
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            accept_multiple_files=True,
            key="image_uploader2"
        )
        
        if training_masks:
            st.session_state.training_masks.extend(training_masks)
            st.session_state.show_uploader2 = False
            st.rerun()
    
    # Display masks in carousel if any exist
    if st.session_state.training_masks:
        st.success(f"{len(st.session_state.training_masks)} masks(s) loaded")
        
        # Create a carousel with arrows to navigate
        cols = st.columns([1, 4, 1])
        with cols[1]:
            if 'current_image_index' not in st.session_state:
                st.session_state.current_image_index2 = 0
            
            current_img = st.session_state.training_masks[st.session_state.current_image_index2]
            st.image(
                current_img,
                caption=f"Image {st.session_state.current_image_index2 + 1}/{len(st.session_state.training_masks)}: {current_img.name}",
                use_container_width=True
            )
        
        # Navigation arrows
        with cols[0]:
            if st.button("<--"):
                st.session_state.current_image_index2 = max(0, st.session_state.current_image_index2 - 1)
                st.rerun()
        
        with cols[2]:
            if st.button("-->"):
                st.session_state.current_image_index2 = min(len(st.session_state.training_masks) - 1, st.session_state.current_image_index2 + 1)
                st.rerun()
        
        # Clear all button
        if st.button("Remove All Masks"):
            st.session_state.training_masks = []
            if 'current_image_index' in st.session_state:
                del st.session_state.current_image_index2
            st.rerun() # Refresh

def test():
    st.title("Test")

def metrics():
    st.title("Metrics")

def images():
    st.title("Images")

# Sidebar 
with st.sidebar:
    st.header("Navigation")
    if st.button("Home"):
        st.session_state.current_screen = "Home"
    if st.button("Train"):
        st.session_state.current_screen = "Train"
    if st.button("Test"):
        st.session_state.current_screen = "Test"
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
elif st.session_state.current_screen == "Train":
    train()
elif st.session_state.current_screen == "Test":
    test()