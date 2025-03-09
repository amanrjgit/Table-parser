# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 01:02:54 2025

@author: Aman Jaiswar
"""

import torch
import json
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import easyocr
import cv2

def load_table_detection_model():
    """
    Load the pre-trained table detection model.

    Returns:
    tuple: Model and feature extractor
    """
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/table-transformer-detection")

    return model, feature_extractor

def preprocess_image(image_path):
    """
    Preprocess the input image for table detection.

    Args:
    image_path (str): Path to the input image

    Returns:
    PIL.Image: Processed image
    """
    # Open the image
    image = Image.open(image_path).convert("RGB")

    return image

def detect_table_regions(image, model, feature_extractor):
    """
    Detect table regions in the image.

    Args:
    image (PIL.Image): Input image
    model (AutoModelForObjectDetection): Trained detection model
    feature_extractor (AutoFeatureExtractor): Feature extractor

    Returns:
    dict: Detected table regions
    """
    # Prepare inputs for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process detection results
    target_sizes = torch.tensor([image.size[::-1]])

    # Fallback processing
    processed_results = {
        'boxes': [],
        'scores': [],
        'labels': []
    }

    try:
        results = feature_extractor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]
        processed_results = results
    except Exception as e:
        print(f"Error processing detection results: {e}")
        # Use entire image if no specific region detected
        processed_results['boxes'] = torch.tensor([[0, 0, image.width, image.height]])

    return processed_results

def extract_table_from_image(image, table_region):
    """
    Extract the table region from the original image.

    Args:
    image (PIL.Image): Original image
    table_region (dict): Detected table region

    Returns:
    numpy.ndarray: Cropped table image
    """
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Extract coordinates
    try:
        # Handle tensor-based boxes
        if isinstance(table_region['boxes'], torch.Tensor):
            x_min, y_min, x_max, y_max = (
                int(table_region['boxes'][0][0].item()),
                int(table_region['boxes'][0][1].item()),
                int(table_region['boxes'][0][2].item()),
                int(table_region['boxes'][0][3].item())
            )
        else:
            # Fallback to previous method
            x_min, y_min, x_max, y_max = (
                int(table_region['boxes'][0][0]),
                int(table_region['boxes'][0][1]),
                int(table_region['boxes'][0][2]),
                int(table_region['boxes'][0][3])
            )
    except (IndexError, KeyError):
        # Fallback to entire image
        height, width = img_cv.shape[:2]
        x_min, y_min, x_max, y_max = 0, 0, width, height

    # Crop the table region
    table_img = img_cv[y_min:y_max, x_min:x_max]

    return table_img

def perform_ocr(table_image):
    """
    Perform OCR on the table image to extract text using EasyOCR.

    Args:
    table_image (numpy.ndarray): Cropped table image

    Returns:
    str: Extracted text from the table
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Perform OCR
    results = reader.readtext(table_image)

    # Extract text from results
    text_lines =  [result[1] for result in results]

    return '\n'.join(text_lines)

def parse_table_text(ocr_text):
    """
    Parse the OCR text into a structured format.

    Args:
    ocr_text (str): Text extracted from the table

    Returns:
    dict: Parsed table data
    """
    # Split text into lines
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]

    # Try to identify headers (first line or first row)
    headers = lines[0].split() if lines else []

    # Parse data rows
    data = []
    for line in lines[1:]:
        row_data = line.split()
        if row_data and len(row_data) == len(headers):
            data.append(dict(zip(headers, row_data)))

    return {
        'headers': headers,
        'data': data,
        'raw_text': ocr_text
    }

def convert_table_to_json(parsed_data, output_path):
    """
    Convert parsed table data to JSON file.

    Args:
    parsed_data (dict): Parsed table data
    output_path (str): Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(parsed_data, f, indent=2)
        
def process_ocr_to_json_financial(ocr_results, row_threshold=10, column_threshold=10):
    """
    Process EasyOCR results into a structured financial statement JSON similar to the provided format.
    Handles tables with or without serial number columns.
    
    Args:
        ocr_results: List of tuples from EasyOCR (bbox, text, confidence)
        row_threshold: Vertical distance threshold to consider texts in the same row (pixels)
        column_threshold: Horizontal distance threshold to consider texts in the same column (pixels)
        
    Returns:
        Dict containing structured financial data
    """
    # Step 1: Extract key information and calculate center points
    processed_items = []
    for bbox, text, confidence in ocr_results:
        # Calculate center point of bounding box
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Get top-left corner for initial sorting
        top_left_y = min(y_coords)
        top_left_x = min(x_coords)
        
        processed_items.append({
            'bbox': bbox,
            'text': text,
            'confidence': confidence,
            'center_x': center_x,
            'center_y': center_y,
            'top_left_y': top_left_y,
            'top_left_x': top_left_x
        })
    
    # Step 2: Group items by rows based on y-coordinate proximity
    processed_items.sort(key=lambda x: x['top_left_y'])
    
    rows = []
    current_row = [processed_items[0]]
    
    for item in processed_items[1:]:
        # If this item is approximately on the same line as previous item
        if abs(item['top_left_y'] - current_row[0]['top_left_y']) <= row_threshold:
            current_row.append(item)
        else:
            # Sort the completed row by x-coordinate (left to right)
            current_row.sort(key=lambda x: x['top_left_x'])
            rows.append(current_row)
            current_row = [item]
    
    # Don't forget to add the last row
    if current_row:
        current_row.sort(key=lambda x: x['top_left_x'])
        rows.append(current_row)
    
    # Step 3: Detect if there's a serial number column
    has_serial_column = False
    serial_column_index = 0
    
    # Check if first column of data rows contains sequential numbers
    number_pattern = []
    for row_idx in range(min(5, len(rows))):
        if len(rows[row_idx]) > 0:
            text = rows[row_idx][0]['text'].strip()
            # Check if it's a number or could be a serial number format (like "1." or "1)")
            if text.isdigit() or (len(text) > 1 and text[:-1].isdigit() and text[-1] in '.):'):
                number_pattern.append(text)
    
    # If we have sequential numbers in the first column
    if len(number_pattern) >= 3:
        try:
            # Extract numbers from patterns like "1." or "1)"
            cleaned_numbers = []
            for num in number_pattern:
                if num.isdigit():
                    cleaned_numbers.append(int(num))
                else:
                    # Extract digit part
                    digit_part = ''.join(c for c in num if c.isdigit())
                    if digit_part:
                        cleaned_numbers.append(int(digit_part))
            
            # Check if they're sequential
            is_sequential = all(cleaned_numbers[i] + 1 == cleaned_numbers[i+1] for i in range(len(cleaned_numbers)-1))
            
            if is_sequential:
                has_serial_column = True
                serial_column_index = 0
        except (ValueError, IndexError):
            # If we can't process the patterns, assume no serial column
            pass
    
    # Step 4: Identify headers and periods from the first few rows
    # We'll assume the first row might contain title/statement type
    # Second row might contain column headers (periods)
    
    # Extract potential report title
    report_title = rows[0][0]['text'] if len(rows) > 0 and len(rows[0]) > 0 else "Financial Results"
    
    # Determine if the table is likely standalone or consolidated based on text
    is_consolidated = False
    if any("consolidated" in item['text'].lower() for row in rows[:3] for item in row):
        is_consolidated = True
    
    # Extract period headers from the first few rows
    period_headers = []
    for row_idx in range(min(3, len(rows))):
        for item in rows[row_idx]:
            # Look for typical period text patterns
            text = item['text'].lower()
            if any(keyword in text for keyword in ["quarter", "year", "month", "ended", "31", "30", "march", "june", "september", "december"]):
                # Clean up the period text
                period_text = item['text'].strip()
                if period_text and period_text not in period_headers:
                    period_headers.append(period_text)
    
    # If we couldn't find period headers, use generic ones based on structure
    if not period_headers and len(rows[0]) > 1:
        period_headers = [f"Period {i+1}" for i in range(len(rows[0])-1)]
    
    # Step 5: Identify line items from the appropriate column (accounting for serial numbers)
    line_item_column = serial_column_index + 1 if has_serial_column else 0
    line_items = []
    
    # Start from a row after headers
    start_row = min(3, len(rows))
    for row_idx in range(start_row, len(rows)):
        if len(rows[row_idx]) > line_item_column:
            # Get the line item text from the correct column
            line_item_text = rows[row_idx][line_item_column]['text'].strip()
            # Filter out non-line items like headers, etc.
            if line_item_text and not any(keyword in line_item_text.lower() for keyword in period_headers):
                line_items.append(line_item_text)
    
    # Step 6: Build the financial statements structure
    result = {}
    
    # Determine the appropriate section name
    if "balance sheet" in report_title.lower():
        section_name = "Balance_sheet"
    elif "cash flow" in report_title.lower():
        section_name = "Cash_flow_statements"
    elif is_consolidated:
        section_name = "Statement_Consolidated_finanacial_results_for_all_months"
    else:
        section_name = "Standalone_financial_results_for_all_months"
    
    # Initialize the section
    if section_name in ["Balance_sheet", "Cash_flow_statements"]:
        # These sections appear to be strings in your example
        result[section_name] = f"{section_name}_are_not_present"
    else:
        result[section_name] = {}
        
        # Calculate column offset for values based on whether we have serial numbers
        value_column_offset = line_item_column + 1
        
        # Extract data for each period
        for period_idx, period in enumerate(period_headers):
            if period_idx < len(period_headers):
                result[section_name][period] = {}
                
                # Process each line item row
                for row_idx in range(start_row, len(rows)):
                    row = rows[row_idx]
                    if len(row) > value_column_offset + period_idx:
                        # Check if this row has a valid line item
                        if line_item_column < len(row):
                            line_item = row[line_item_column]['text'].strip()
                            
                            # Filter out any obvious non-line items (often empty or just punctuation)
                            if len(line_item) > 1 and line_item in line_items:
                                # Get the value for this line item and period
                                value_text = row[value_column_offset + period_idx]['text'].strip()
                                
                                # Try to convert to number if possible
                                try:
                                    # Handle common number formatting issues
                                    value_text = value_text.replace(',', '')
                                    # Handle parentheses for negative numbers, e.g., (123.45)
                                    if value_text.startswith('(') and value_text.endswith(')'):
                                        value_text = '-' + value_text[1:-1]
                                    value = float(value_text)
                                except ValueError:
                                    value = value_text
                                
                                # Add to the result
                                result[section_name][period][line_item] = value
    
    # Add other standard sections from your template if they don't exist
    for section in ["Standalone_financial_results_for_all_months", 
                   "Balance_sheet", 
                   "Cash_flow_statements", 
                   "Statement_Consolidated_finanacial_results_for_all_months"]:
        if section not in result:
            if section in ["Balance_sheet", "Cash_flow_statements"]:
                result[section] = f"{section}_are_not_present"
            else:
                result[section] = {}
    
    return result

def process_table_image(image_path, output_path=None):
    """
    Main function to process table image and convert to JSON.

    Args:
    image_path (str): Path to input image
    output_path (str, optional): Path to save output JSON. If None, returns the parsed data without saving.

    Returns:
    dict: Parsed table data
    """
    try:
        # Load detection model
        model, feature_extractor = load_table_detection_model()

        # Preprocess image
        image = preprocess_image(image_path)

        # Detect table regions
        table_regions = detect_table_regions(image, model, feature_extractor)

        # Extract table from image
        table_image = extract_table_from_image(image, table_regions)

        # Perform OCR
        reader = easyocr.Reader(['en'])
        ocr_results = reader.readtext(table_image)

        # Process OCR results into structured format
        parsed_data = process_ocr_to_json_financial(ocr_results)

        # If output path is provided, save the JSON file
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(parsed_data, f, indent=2)

        return parsed_data
        
    except Exception as e:
        # Handle exceptions more gracefully
        print(f"Error in process_table_image: {str(e)}")
        # Return a minimal structure with error info
        return {
            "error": str(e),
            "status": "failed",
            "message": "Could not process table image"
        }