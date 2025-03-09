# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 23:28:25 2025

@author: Aman Jaiswar
"""

import streamlit as st
import torch
import os
import json
import tempfile
import base64
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
import time
from ocr import process_table_image
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

def main():
    st.set_page_config(page_title="Table Extractor", layout="wide")

    @st.cache_resource
    def load_table_detection_model():
        """
        Load the pre-trained table detection model (cached to save memory).
        """
        try:
            CACHE_DIR = "./model_cache"
            with st.spinner("Downloading table detection model... This may take a minute."):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                st.write(f"Using device: {device}")
                model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", cache_dir=CACHE_DIR)
                st.write("Model loaded successfully")
                model = model.to(device)
                st.write("Model moved to device")
                feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/table-transformer-detection", cache_dir=CACHE_DIR)
                st.write("Feature extractor loaded successfully")
            return model, feature_extractor
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    try:
        st.write("About to load model...")
        model, feature_extractor = load_table_detection_model()
        # Store in session state
        st.session_state.model = model
        st.session_state.feature_extractor = feature_extractor
        st.write("Model loaded successfully and stored in session state")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
    }
    .image-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
    }
    .image-container:hover {
        border-color: #4CAF50;
    }
    .selected {
        border: 3px solid #4CAF50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Document Table Extractor")
    st.subheader("Upload a PDF or Image to Extract Tables")
    
    # Initialize session state for storing image paths
    if 'image_list' not in st.session_state:
        st.session_state.image_list = []
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == ".pdf":
            process_pdf(uploaded_file)
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            process_image(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a PDF or image file.")

def extract_images_from_page(doc, page_num, temp_dir):
    """Extract images from a PDF page and save them to the output directory."""
    image_list = []
    
    # Get the page
    page = doc.load_page(page_num)
    
    # Get image list
    images = page.get_images(full=True)
    
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        
        # Save the image
        image_filename = f"page{page_num+1}_img{img_index+1}.png"
        image_path = os.path.join(temp_dir, image_filename)
        
        with open(image_path, "wb") as img_file:
            img_file.write(image_bytes)
        
        # Add to our image list
        image_list.append({
            "path": image_path,
            "page": page_num + 1,
            "index": img_index + 1,
            "filename": image_filename
        })
    
    return image_list

def process_pdf(pdf_file):
    # Create a persistent temp directory path
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    temp_dir = st.session_state.temp_dir
    
    # Save the PDF to a temporary file
    temp_pdf_path = os.path.join(temp_dir, "uploaded.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Clear previous image list
    st.session_state.image_list = []
    
    try:
        # Open the PDF with PyMuPDF
        doc = fitz.open(temp_pdf_path)
        
        # Extract images from the PDF
        for page_num in range(len(doc)):
            st.session_state.image_list.extend(extract_images_from_page(doc, page_num, temp_dir))
        
        # Close the document to free up resources
        doc.close()
        
        if not st.session_state.image_list:
            st.warning("No images found in the PDF. Try uploading an image directly.")
            return
        
        st.success(f"Found {len(st.session_state.image_list)} images in the PDF.")
        
        # Allow user to select images
        display_image_selection()
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

def process_image(image_file):
    # Create a persistent temp directory path
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    temp_dir = st.session_state.temp_dir
    
    # Save the uploaded image
    image_path = os.path.join(temp_dir, "uploaded_image.png")
    with open(image_path, "wb") as f:
        f.write(image_file.getbuffer())
    
    # Create an image list with just this one image
    st.session_state.image_list = [{
        "path": image_path,
        "page": 1,
        "index": 1,
        "filename": image_file.name
    }]
    
    # Display the image for selection
    display_image_selection()

def display_image_selection():
    """Display images and handle selection logic."""
    st.subheader("Select images to extract tables from:")
    
    # Use session state to track selected images
    if 'selected_indices' not in st.session_state:
        st.session_state.selected_indices = []
    
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = []
    
    # Create a multi-column layout
    cols = 5
    rows = (len(st.session_state.image_list) + cols - 1) // cols
    
    # Use markdown and HTML/CSS for a nicer grid
    st.write("Click on images to select them:")
    
    st.session_state.selected_images = []
    
    for i in range(rows):
        cols_container = st.columns(cols)
        for j in range(cols):
            idx = i * cols + j
            if idx < len(st.session_state.image_list):
                img_info = st.session_state.image_list[idx]
                with cols_container[j]:
                    try:
                        # Display image
                        image = Image.open(img_info["path"])
                        
                        # Create a unique key for this image
                        key = f"img_{img_info['page']}_{img_info['index']}"
                        
                        # Check if this image is selected
                        is_selected = idx in st.session_state.selected_indices
                        
                        # Display selection state
                        selection_class = "selected" if is_selected else ""
                        
                        # Display image with clickable functionality
                        st.markdown(f"""
                        <div class="image-container {selection_class}" id="{key}_container">
                            <p>Page {img_info['page']}, Image {img_info['index']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show the image
                        st.image(image, use_column_width=True)
                        
                        # Checkbox for selection
                        if st.checkbox("Select", key=key, value=is_selected):
                            if idx not in st.session_state.selected_indices:
                                st.session_state.selected_indices.append(idx)
                            # Add to selected images
                            st.session_state.selected_images.append(img_info)
                        else:
                            if idx in st.session_state.selected_indices:
                                st.session_state.selected_indices.remove(idx)
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
    
    # Extract table button
    if st.session_state.selected_images:
        if st.button("Extract Tables"):
            extract_tables(st.session_state.selected_images)
    else:
        st.info("Please select at least one image to extract tables.")

def extract_tables(selected_images):
    """Extract tables from selected images using OCR."""
    st.subheader("Extracting Tables...")

    # Get model and feature_extractor from session state
    model = st.session_state.model
    feature_extractor = st.session_state.feature_extractor
    
    # Progress bar for better UX
    progress_bar = st.progress(0)
    
    # List to store extracted data
    extracted_data = []
    
    for i, img_info in enumerate(selected_images):
        progress_bar.progress((i + 1) / len(selected_images))
        
        try:
            # Call process_table_image from ocr.py
            st.text(f"Processing {img_info['filename']}...")
            image_path = img_info['path']
            
            # Process the image and get structured data
            # We don't need an output path as we'll store in memory for display
            parsed_data = process_table_image(image_path, model, feature_extractor)
            
            # Add image source info to the parsed data
            result = {
                "image_source": f"Page {img_info['page']}, Image {img_info['index']}",
                "filename": img_info['filename'],
                "parsed_data": parsed_data
            }
            
            extracted_data.append(result)
            st.success(f"Successfully processed {img_info['filename']}")
            
        except Exception as e:
            st.error(f"Error extracting data from {img_info['filename']}: {str(e)}")
        
        # Add a slight delay to see the progress
        time.sleep(0.5)
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    
    if not extracted_data:
        st.error("No data could be extracted from the selected images.")
        return
    
    # Create a JSON file with the extracted data
    json_data = json.dumps(extracted_data, indent=2)
    
    # Provide download button
    st.success("Table extraction complete!")
    st.subheader("Download Extracted Tables")
    
    # Create a download button for the JSON
    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name="extracted_tables.json",
        mime="application/json",
    )
    
    # Also display the extracted data
    st.subheader("Extracted Data Preview")
    st.json(json_data)

if __name__ == "__main__":
    main()
