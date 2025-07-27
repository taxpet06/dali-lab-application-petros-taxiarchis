import streamlit as st
from PIL import Image
import io
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import your modified demo functions
from model_training import (
    train_barnacle_model_streamlit, 
    save_model_streamlit, 
    load_model_streamlit,
    predict_barnacles_from_uploaded
)

# Initialize
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Home"

# Initialize model storage
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "model_device" not in st.session_state:
    st.session_state.model_device = None

def home():
    st.title("Barnacle Segmentation App")
    st.markdown("""
        **Instructions**:
        
        1.  Navigate to the ***Train*** tab
        2.  Load your images and masks accordingly  
                - Both images and masks do not need to be loaded in proper order as long as their filename ends with a number (e.g., img1.png, mask1.png)  
                - You may use the ***Remove all images*** or ***Remove all masks*** buttons if a mistake is made
        3.  Configure training parameters and use the ***Train*** button to train the model
        4.  Save the trained model using the ***Save Model*** button
        5.  Navigate to the ***Test*** tab
        6.  Load either the model that was just trained or upload a pre-trained model of your choice
        7.  Load testing data
        8.  Use the ***Segment*** button to perform barnacle classification
        9.  The *Metrics* tab will display a variety of useful information and graphs regarding your training and testing datasets
        10. The ***Images*** tab will display the images and masks for your training and testing datasets
        """
    )

def train():
    st.title("Train Barnacle Segmentation Model")
    
    #---Images---#
    # Initialization of list
    if 'training_images' not in st.session_state:
        st.session_state.training_images = []
    
    # Button to trigger file uploader
    if st.button("Load Image(s)"):
        st.session_state.show_uploader = True
    
    # File uploader
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
            st.rerun()

    #---Masks---#        
    # Initialization of list
    if 'training_masks' not in st.session_state:
        st.session_state.training_masks = []
    
    # Button to trigger file uploader
    if st.button("Load Mask(s)"):
        st.session_state.show_uploader2 = True
    
    # File uploader
    if st.session_state.get('show_uploader2', False):
        training_masks = st.file_uploader(
            "Choose masks", 
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
        st.success(f"{len(st.session_state.training_masks)} mask(s) loaded")
        
        # Create a carousel with arrows to navigate
        cols = st.columns([1, 4, 1])
        with cols[1]:
            if 'current_image_index2' not in st.session_state:
                st.session_state.current_image_index2 = 0
            
            current_img = st.session_state.training_masks[st.session_state.current_image_index2]
            st.image(
                current_img,
                caption=f"Mask {st.session_state.current_image_index2 + 1}/{len(st.session_state.training_masks)}: {current_img.name}",
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
            if 'current_image_index2' in st.session_state:
                del st.session_state.current_image_index2
            st.rerun()
    
    #---Training Configuration---#
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        num_epochs = st.slider("Number of Epochs", min_value=10, max_value=200, value=50, step=10) # slider to pick epochs
    with col2:
        learning_rate = st.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], index=2, format_func=lambda x: f"{x:.0e}") # pick learning rate
    
    #---Training Button---#
    if st.button("Train Model", type="primary"):
        if not st.session_state.training_images or not st.session_state.training_masks:
            st.error("Please load both training images and masks before training.") # print message if lack of masks/images
        elif len(st.session_state.training_images) != len(st.session_state.training_masks): # print error for wrong amount of masks/images
            st.error(f"Number of images ({len(st.session_state.training_images)}) must match number of masks ({len(st.session_state.training_masks)})")
        else:
            # Create progressbar
            progress_container = st.container()
            
            try:
                with st.spinner("Training model..."): # progressbar
                    # Train the model using additional python file
                    trained_model, device = train_barnacle_model_streamlit(
                        st.session_state.training_images,
                        st.session_state.training_masks,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate,
                        progress_container=progress_container
                    )
                    
                    # Save trained model 
                    st.session_state.trained_model = trained_model
                    st.session_state.model_device = device
                    
                    st.success("Training completed successfully!")
                    
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
    
    #---Save Model---#
    if st.session_state.trained_model is not None:
        st.subheader("Save Trained Model")
        
        model_filename = st.text_input("Model filename", value="barnacle_mobilesam.pth") # give option for model renaming
        
        if st.button("Save Model"):
            try:
                saved_path = save_model_streamlit(st.session_state.trained_model, model_filename) # save model
                st.success(f"Model saved as: {saved_path}")
                
                # Provide download button for user
                with open(saved_path, "rb") as file:
                    st.download_button(
                        label="Download Model File",
                        data=file.read(),
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )
            except Exception as e:
                st.error(f"Failed to save model: {str(e)}")

def test():
    st.title("Test Barnacle Segmentation")
    
    #Model Loading
    st.subheader("Load Model")
    
    model_source = st.radio("Model Source", ["Use Trained Model", "Upload Model File"]) # choose between trained model or one the user loads
    
    loaded_model = None
    device = None
    
    if model_source == "Use Trained Model": # for use trained model, load the recently trained model
        if st.session_state.trained_model is not None:
            loaded_model = st.session_state.trained_model
            device = st.session_state.model_device
            st.success("Using currently trained model")
        else: # throw error if model has not been trained yet
            st.warning("No trained model available. Please train a model first or upload a model file.")
    
    else:  # Upload Model File
        uploaded_model = st.file_uploader("Choose model file", type=["pth", "pt"]) # allow for model upload
        if uploaded_model is not None:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                loaded_model = load_model_streamlit(uploaded_model, device) # and load
                st.success(f"Model loaded successfully on {device}")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
    
    # Test Image Upload
    if loaded_model is not None:
        st.subheader("Test Images")
        
        test_images = st.file_uploader( # allow user to upload test images
            "Choose test images",
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            accept_multiple_files=True
        )
        
        if test_images:
            st.success(f"{len(test_images)} test image(s) loaded")
            
            # allow user to select which image they are previewing
            if len(test_images) > 1:
                selected_idx = st.selectbox("Select image to preview", range(len(test_images)), 
                                          format_func=lambda x: test_images[x].name)
            else:
                selected_idx = 0
            
            # Display selected image for preview
            st.image(test_images[selected_idx], caption=f"Test Image: {test_images[selected_idx].name}", use_container_width=True)
            
            # Segmentation
            if st.button("Segment Barnacles", type="primary"):
                results_container = st.container()
                
                with st.spinner("Performing segmentation..."): # progress bar
                    try:
                        for i, test_image in enumerate(test_images):
                            # Predict barnacles
                            predicted_mask, original_image = predict_barnacles_from_uploaded(
                                loaded_model, test_image, device
                            )
                            
                            # Post-process to get circular outlines
                            kernel = np.ones((3,3), np.uint8)
                            cleaned_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_OPEN, kernel)
                            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
                            
                            # Find contours for barnacle outlines
                            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            # Create outline image
                            outline_image = original_image.copy()
                            cv2.drawContours(outline_image, contours, -1, (0, 255, 0), 2)
                            
                            # Display results
                            results_container.subheader(f"Results for {test_image.name}")
                            
                            col1, col2 = results_container.columns(2)
                            
                            with col1:
                                st.image(original_image, caption="Original Image", use_container_width=True)
                            with col2:
                                st.image(outline_image, caption=f"Detected Barnacles ({len(contours)} found)", use_container_width=True)
                            
                            results_container.success(f"Detected {len(contours)} potential barnacles in {test_image.name}")
                            
                            # Store results in session state for metrics tab
                            if 'test_results' not in st.session_state:
                                st.session_state.test_results = []
                            
                            st.session_state.test_results.append({ # keep test results for each image
                                'image_name': test_image.name,
                                'original_image': original_image,
                                'predicted_mask': predicted_mask,
                                'outline_image': outline_image,
                                'barnacle_count': len(contours),
                                'contours': contours
                            })
                        
                        st.success("Segmentation completed for all images!")
                        
                    except Exception as e:
                        st.error(f"Segmentation failed: {str(e)}")

def metrics():
    st.title("Metrics and Analysis")
    
    # Training Data Metrics
    if (hasattr(st.session_state, 'training_images') and st.session_state.training_images and 
        hasattr(st.session_state, 'training_masks') and st.session_state.training_masks):
        st.subheader("Training Data")
        
        col1, col2 = st.columns(2) # Quick counts for images and masks in training
        with col1:
            st.metric("Number of Training Images", len(st.session_state.training_images))
        with col2:
            st.metric("Number of Training Masks", len(st.session_state.training_masks))
        
        # Display file names and sizes
        st.write("**Training Image Files:**")
        for i, img in enumerate(st.session_state.training_images):
            st.write(f"{i+1}. {img.name} ({img.size} bytes)")
        
        st.write("**Training Mask Files:**")
        for i, mask in enumerate(st.session_state.training_masks):
            st.write(f"{i+1}. {mask.name} ({mask.size} bytes)")
    
    # Test Results Metrics
    if 'test_results' in st.session_state and st.session_state.test_results:
        st.subheader("Test Results")
        
        # Calculate images and predicted and average barnacles
        total_images = len(st.session_state.test_results)
        total_barnacles = sum(result['barnacle_count'] for result in st.session_state.test_results)
        avg_barnacles = total_barnacles / total_images if total_images > 0 else 0
        
        col1, col2, col3 = st.columns(3) # Easy metrics
        with col1:
            st.metric("Total Test Images", total_images)
        with col2:
            st.metric("Total Barnacles Detected", total_barnacles)
        with col3:
            st.metric("Average Barnacles per Image", f"{avg_barnacles:.1f}")
        
        # Barnacle count distribution chart
        barnacle_counts = [result['barnacle_count'] for result in st.session_state.test_results]
        
        if len(barnacle_counts) > 1:
            # Set up bar chart
            st.subheader("Barnacle Count Distribution")
            import pandas as pd
            df = pd.DataFrame({
                'Image': [result['image_name'] for result in st.session_state.test_results],
                'Barnacle Count': barnacle_counts
            })
            
            # Bar chart
            st.bar_chart(df.set_index('Image'))
            
            # Statistics
            st.write("**Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Barnacles", min(barnacle_counts))
            with col2:
                st.metric("Max Barnacles", max(barnacle_counts))
            with col3:
                st.metric("Median Barnacles", np.median(barnacle_counts))
            with col4:
                st.metric("Std Deviation", f"{np.std(barnacle_counts):.1f}")
        
        # quick table
        st.subheader("Detailed Results")
        results_df = pd.DataFrame([
            {
                'Image Name': result['image_name'],
                'Barnacles Detected': result['barnacle_count'],
                'Image Size (pixels)': f"{result['original_image'].shape[1]} x {result['original_image'].shape[0]}"
            }
            for result in st.session_state.test_results
        ])
        st.dataframe(results_df, use_container_width=True)
    
    else:
        st.info("No test results available. Please run segmentation on test images first.")

def images():
    st.title("Image Gallery")
    
    # Training Images
    if (hasattr(st.session_state, 'training_images') and st.session_state.training_images and 
        hasattr(st.session_state, 'training_masks') and st.session_state.training_masks):
        st.subheader("Training Data")
        
        # Create tabs for training images and masks
        tab1, tab2 = st.tabs(["Training Images", "Training Masks"])
        
        with tab1: 
            if hasattr(st.session_state, 'training_images') and st.session_state.training_images:
                cols = st.columns(min(3, len(st.session_state.training_images)))
                for i, img in enumerate(st.session_state.training_images):
                    with cols[i % 3]:
                        st.image(img, caption=f"{img.name}", use_container_width=True)
        
        with tab2:
            if hasattr(st.session_state, 'training_masks') and st.session_state.training_masks:
                cols = st.columns(min(3, len(st.session_state.training_masks)))
                for i, mask in enumerate(st.session_state.training_masks):
                    with cols[i % 3]:
                        st.image(mask, caption=f"{mask.name}", use_container_width=True)
    
    # Test Images
    if 'test_results' in st.session_state and st.session_state.test_results:
        st.subheader("Testing Data")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Original Images", "Segmentation Results"])
        
        with tab1: # display image
            cols = st.columns(min(3, len(st.session_state.test_results)))
            for i, result in enumerate(st.session_state.test_results):
                with cols[i % 3]:
                    st.image(result['original_image'], caption=result['image_name'], use_container_width=True)
        with tab2: # display segmentataion
            cols = st.columns(min(3, len(st.session_state.test_results)))
            for i, result in enumerate(st.session_state.test_results):
                with cols[i % 3]:
                    st.image(result['outline_image'], caption=f"{result['barnacle_count']} barnacles - {result['image_name']}", use_container_width=True)
    
    # Clear results button
    if 'test_results' in st.session_state and st.session_state.test_results:
        if st.button("Clear Test Results"):
            del st.session_state.test_results
            st.rerun()

# Sidebar Navigation
with st.sidebar:
    st.header("Barnacle Segmentation using MobileSam")
    st.markdown("---")
    if st.button("Home", use_container_width=True):
        st.session_state.current_screen = "Home"
    if st.button("Train", use_container_width=True):
        st.session_state.current_screen = "Train"
    if st.button("Test", use_container_width=True):
        st.session_state.current_screen = "Test"
    if st.button("Metrics", use_container_width=True):
        st.session_state.current_screen = "Metrics"
    if st.button("Images", use_container_width=True):
        st.session_state.current_screen = "Images"
    st.markdown("---")
    
    # System Info
    st.subheader("System Info")
    device_info = "CUDA Available" if torch.cuda.is_available() else "CPU Only"
    st.info(f"Device: {device_info}")
    
    if st.session_state.trained_model is not None:
        st.success("Model Trained")
    else:
        st.warning("No Trained Model")
    st.markdown("---")
    # Quick Stats
    st.subheader("Quick Statistics")
    if hasattr(st.session_state, 'training_images') and st.session_state.training_images:
        st.metric("Training Images", len(st.session_state.training_images))
    if hasattr(st.session_state, 'training_masks') and st.session_state.training_masks:
        st.metric("Training Masks", len(st.session_state.training_masks))
    
    if 'test_results' in st.session_state and st.session_state.test_results:
        total_barnacles = sum(result['barnacle_count'] for result in st.session_state.test_results) # Print total predicted barnalces in test data
        st.metric("Total Barnacles Found", total_barnacles)

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