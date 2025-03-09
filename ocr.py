# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 01:02:54 2025

@author: Aman Jaiswar
"""

import torch
import json
import numpy as np
from PIL import Image
import easyocr
import cv2
import streamlit as st
import os

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

def process_ocr_to_json_financial(ocr_results, row_threshold=10, column_threshold=10):
    """
    Process EasyOCR results into a structured financial statement JSON.
    Handles various financial statement formats with different time periods and line items.
    
    Args:
        ocr_results: List of tuples from EasyOCR (bbox, text, confidence)
        row_threshold: Vertical distance threshold to consider texts in the same row (pixels)
        column_threshold: Horizontal distance threshold to consider texts in the same column (pixels)
        
    Returns:
        Dict containing structured financial data in the standardized format
    """
    # Step 1: Extract key information and calculate center points
    processed_items = []
    for bbox, text, confidence in ocr_results:
        # Calculate center point and dimensions of bounding box
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Get top-left corner for initial sorting
        top_left_y = min(y_coords)
        top_left_x = min(x_coords)
        
        # Get width and height for later analysis
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        processed_items.append({
            'bbox': bbox,
            'text': text.strip(),
            'confidence': confidence,
            'center_x': center_x,
            'center_y': center_y,
            'top_left_y': top_left_y,
            'top_left_x': top_left_x,
            'width': width,
            'height': height
        })
    
    # Step 2: Group items by rows based on y-coordinate proximity
    processed_items.sort(key=lambda x: x['top_left_y'])
    
    rows = []
    if not processed_items:
        return create_empty_result()
        
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
    
    # Step 3: Identify the type of report and its structure
    # Analyze the first few rows to determine the report type and structure
    report_info = identify_report_type(rows[:10])
    report_type = report_info['report_type']
    title = report_info['title']
    is_consolidated = report_info['is_consolidated']
    
    # Step 4: Identify the time periods (columns) from the headers
    time_periods = identify_time_periods(rows[:10])
    
    # If no explicit time periods found, try to detect them from column structure
    if not time_periods:
        time_periods = detect_implicit_time_periods(rows)
    
    # Step 5: Identify the line items and data structure
    data_structure = identify_data_structure(rows, time_periods)
    
    # Step 6: Extract the financial data
    financial_data = extract_financial_data(rows, data_structure, time_periods)
    
    # Step 7: Format the data according to the standardized structure
    result = format_result(financial_data, report_type, is_consolidated)
    
    return result

def identify_report_type(header_rows):
    """
    Identify the type of financial report from the header rows.
    
    Args:
        header_rows: List of rows containing header information
        
    Returns:
        Dict containing report type, title, and consolidation status
    """
    title = ""
    is_consolidated = False
    report_type = "income_statement"  # Default type
    
    # Extract text from header rows and join
    header_text = " ".join([item['text'].lower() for row in header_rows for item in row])
    
    # Check for consolidated statement
    if any(term in header_text for term in ["consolidated", "group", "and its subsidiaries"]):
        is_consolidated = True
    
    # Check for balance sheet
    if any(term in header_text for term in ["balance sheet", "statement of financial position", "assets", "liabilities", "equity"]):
        report_type = "balance_sheet"
    
    # Check for cash flow statement
    elif any(term in header_text for term in ["cash flow", "statement of cash flows", "operating activities", "investing activities", "financing activities"]):
        report_type = "cash_flow"
    
    # Check for income statement / profit & loss
    elif any(term in header_text for term in ["income statement", "profit and loss", "statement of profit", "p&l", "profit/loss", "revenue", "expenses"]):
        report_type = "income_statement"
    
    # Extract potential title from first couple of rows
    for row in header_rows[:2]:
        if row and len(row[0]['text']) > 5:  # Assuming the title has some meaningful length
            title = row[0]['text']
            break
    
    return {
        'report_type': report_type,
        'title': title,
        'is_consolidated': is_consolidated
    }

def identify_time_periods(header_rows):
    """
    Identify time periods from header rows.
    
    Args:
        header_rows: List of rows containing header information
        
    Returns:
        List of dictionaries containing period information
    """
    time_periods = []
    period_keywords = ["quarter", "month", "year", "q1", "q2", "q3", "q4", "ended", "ending"]
    month_keywords = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    
    for row in header_rows:
        for item in row:
            text = item['text'].lower()
            
            # Skip very short texts
            if len(text) < 2:
                continue
                
            # Check if this item contains period-related words
            if any(keyword in text for keyword in period_keywords) or any(month in text for month in month_keywords):
                # Check if the text contains a date format (looking for digits and separators)
                has_date_format = any(c.isdigit() for c in text) and any(c in text for c in ['-', '.', '/', ' '])
                
                if has_date_format or any(keyword in text for keyword in period_keywords):
                    # Clean up the period text
                    period_text = item['text'].strip()
                    period_type = classify_period_type(period_text)
                    
                    if period_text and not any(period_text == p['text'] for p in time_periods):
                        time_periods.append({
                            'text': period_text,
                            'type': period_type,
                            'center_x': item['center_x']
                        })
    
    return time_periods

def classify_period_type(period_text):
    """
    Classify the type of time period (quarter, year, etc.)
    
    Args:
        period_text: Text representing the time period
        
    Returns:
        String representing the period type
    """
    text = period_text.lower()
    
    if any(term in text for term in ["quarter", "3 month", "three month", "q1", "q2", "q3", "q4"]):
        return "Quarter Ended"
    elif any(term in text for term in ["9 month", "nine month"]):
        return "Nine Month Ended"
    elif any(term in text for term in ["12 month", "twelve month", "year", "annual"]):
        return "Year Ended"
    elif any(term in text for term in ["6 month", "six month", "half year"]):
        return "Six Month Ended"
    else:
        # Default to quarter if unclear
        return "Quarter Ended"

def detect_implicit_time_periods(rows):
    """
    Detect time periods based on the structure of data when explicit headers aren't found.
    
    Args:
        rows: All rows of data
        
    Returns:
        List of dictionaries containing period information
    """
    # Look for date patterns in the data columns
    date_patterns = []
    
    # Check the first few rows after potential headers (rows 3-7)
    for row_idx in range(min(3, len(rows)), min(7, len(rows))):
        if len(rows[row_idx]) >= 2:  # Need at least a label and one data point
            # Skip the first column (likely line item labels)
            for col_idx in range(1, len(rows[row_idx])):
                text = rows[row_idx][col_idx]['text']
                # Check if it looks like a date (contains digits and separators)
                if (any(c.isdigit() for c in text) and 
                    any(c in text for c in ['-', '.', '/', ' ']) and
                    len(text) >= 4):  # Minimum length for a reasonable date
                    
                    date_patterns.append({
                        'text': text,
                        'type': "Quarter Ended",  # Default type
                        'center_x': rows[row_idx][col_idx]['center_x']
                    })
                    break  # Found a date in this row
    
    # If we found date patterns, return them
    if date_patterns:
        return date_patterns
    
    # If no date patterns, return generic periods based on the number of data columns
    generic_periods = []
    # Check a row that's likely to contain data (not headers)
    for row_idx in range(min(5, len(rows)), min(10, len(rows))):
        if len(rows[row_idx]) >= 2:  # Need at least a label and one data point
            # Count likely data columns (excluding the first column which is likely labels)
            num_data_cols = len(rows[row_idx]) - 1
            
            for i in range(num_data_cols):
                generic_periods.append({
                    'text': f"Period {i+1}",
                    'type': "Quarter Ended",  # Default type
                    'center_x': rows[row_idx][i+1]['center_x']  # Center X of the data column
                })
            break
    
    return generic_periods

def identify_data_structure(rows, time_periods):
    """
    Identify the structure of the data (line items, columns, etc.)
    
    Args:
        rows: All rows of data
        time_periods: Identified time periods
        
    Returns:
        Dictionary containing data structure information
    """
    # Identify potential rows that contain data (after headers)
    start_row_idx = 0
    
    # Skip the first few rows which likely contain headers
    for idx in range(min(10, len(rows))):
        row_text = " ".join([item['text'].lower() for item in rows[idx]])
        if any(term in row_text for term in ["revenue", "income", "sales", "total", "expenses", "profit"]):
            start_row_idx = idx + 1
            break
    
    # If no clear start row was found, use a default
    if start_row_idx == 0:
        start_row_idx = min(3, len(rows))
    
    # Identify columns and their purposes
    columns = []
    
    # If we have rows to analyze
    if start_row_idx < len(rows) and rows[start_row_idx]:
        # Assuming the first column is for line items
        columns.append({'type': 'line_item', 'center_x': rows[start_row_idx][0]['center_x']})
        
        # Analyze the first data row to identify value columns
        for col_idx in range(1, len(rows[start_row_idx])):
            item = rows[start_row_idx][col_idx]
            text = item['text']
            
            # Check if this column contains numeric data
            is_numeric = is_likely_numeric(text)
            
            if is_numeric:
                # Find the corresponding time period for this column
                matching_period = None
                for period in time_periods:
                    # Match by approximate x-coordinate
                    if abs(period['center_x'] - item['center_x']) < 50:  # Threshold for matching
                        matching_period = period
                        break
                
                if not matching_period and time_periods:
                    # If can't match by position, assign sequentially
                    period_idx = min(col_idx - 1, len(time_periods) - 1)
                    matching_period = time_periods[period_idx]
                
                columns.append({
                    'type': 'value',
                    'center_x': item['center_x'],
                    'period': matching_period['text'] if matching_period else f"Period {col_idx}"
                })
    
    return {
        'start_row_idx': start_row_idx,
        'columns': columns
    }

def is_likely_numeric(text):
    """
    Check if text likely represents a numeric value in financial statements.
    
    Args:
        text: Text to check
        
    Returns:
        Boolean indicating if the text is likely numeric
    """
    # Remove common non-numeric characters
    cleaned_text = text.replace(',', '').replace(' ', '').replace('(', '').replace(')', '')
    
    # Check for negative numbers in parentheses format
    if text.startswith('(') and text.endswith(')'):
        cleaned_text = cleaned_text.replace('(', '').replace(')', '')
    
    # Allow for decimal points
    if '.' in cleaned_text:
        parts = cleaned_text.split('.')
        if len(parts) == 2:
            return parts[0].replace('-', '').isdigit() and parts[1].isdigit()
    
    # Check if it's a simple integer or negative number
    return cleaned_text.replace('-', '').isdigit()

def extract_financial_data(rows, data_structure, time_periods):
    """
    Extract financial data from the OCR results based on the identified structure.
    
    Args:
        rows: All rows of data
        data_structure: Identified data structure
        time_periods: Identified time periods
        
    Returns:
        Dictionary containing extracted financial data
    """
    financial_data = {}
    
    # Initialize structure for each time period
    for period in time_periods:
        period_type = period['type']
        period_text = period['text']
        
        if period_type not in financial_data:
            financial_data[period_type] = {}
        
        financial_data[period_type][period_text] = {}
    
    # Initialize tracking of current section
    current_section = None
    current_subsection = None
    
    # Process rows from the identified start row
    start_row_idx = data_structure['start_row_idx']
    for row_idx in range(start_row_idx, len(rows)):
        row = rows[row_idx]
        
        if not row:
            continue
        
        # Get the item text (usually in the first column)
        if row and len(row) > 0:
            item_text = row[0]['text'].strip()
            
            # Skip empty or very short item texts
            if not item_text or len(item_text) < 2:
                continue
            
            # Check if this is a section header
            if (item_text.isupper() or 
                any(term in item_text.lower() for term in ["total", "subtotal", "INCOME", "EXPENSES", "PROFIT", "LOSS"])):
                if any(term in item_text.lower() for term in ["income", "revenue", "sales"]):
                    current_section = "Income From Operations"
                    current_subsection = None
                elif any(term in item_text.lower() for term in ["expense", "cost"]):
                    current_section = "Expenses"
                    current_subsection = None
                elif any(term in item_text.lower() for term in ["tax"]):
                    current_section = "Tax Expense"
                    current_subsection = None
                elif any(term in item_text.lower() for term in ["comprehensive"]):
                    current_section = "Other Comprehensive Income"
                    current_subsection = None
                elif "earnings per" in item_text.lower():
                    current_section = "Earnings per equity share"
                    current_subsection = None
                continue
            
            # Process data for each time period
            for col_idx in range(1, len(row)):
                if col_idx >= len(row):
                    continue
                    
                value_text = row[col_idx]['text'].strip()
                if not value_text:
                    continue
                
                # Try to find the matching period for this column
                matching_period = find_matching_period(row[col_idx], time_periods, col_idx, len(time_periods))
                
                if not matching_period:
                    continue
                
                period_type = matching_period['type']
                period_text = matching_period['text']
                
                # Convert value to number if possible
                value = parse_financial_value(value_text)
                
                # Store data in the appropriate structure
                if current_section:
                    # If it's a subsection
                    if current_subsection:
                        if current_section not in financial_data[period_type][period_text]:
                            financial_data[period_type][period_text][current_section] = {}
                        
                        if current_subsection not in financial_data[period_type][period_text][current_section]:
                            financial_data[period_type][period_text][current_section][current_subsection] = {}
                        
                        financial_data[period_type][period_text][current_section][current_subsection][item_text] = value
                    else:
                        # It's in a section but not a subsection
                        if current_section not in financial_data[period_type][period_text]:
                            financial_data[period_type][period_text][current_section] = {}
                        
                        financial_data[period_type][period_text][current_section][item_text] = value
                else:
                    # Direct line item (not in a section)
                    financial_data[period_type][period_text][item_text] = value
    
    return financial_data

def find_matching_period(item, time_periods, col_idx, num_periods):
    """
    Find the matching time period for a data column.
    
    Args:
        item: The data item
        time_periods: List of identified time periods
        col_idx: Column index
        num_periods: Number of periods
        
    Returns:
        Matching period dictionary or None
    """
    if not time_periods:
        return None
    
    # First try to match by x-coordinate
    for period in time_periods:
        if abs(period['center_x'] - item['center_x']) < 50:  # Threshold for matching
            return period
    
    # If can't match by position, assign sequentially based on column index
    period_idx = min(col_idx - 1, len(time_periods) - 1)
    if period_idx >= 0 and period_idx < len(time_periods):
        return time_periods[period_idx]
    
    return None

def parse_financial_value(text):
    """
    Parse a financial value from text.
    
    Args:
        text: Text representation of a financial value
        
    Returns:
        Float if parseable, otherwise the original text
    """
    # Clean the text
    cleaned_text = text.replace(',', '')
    
    # Handle parentheses for negative numbers
    is_negative = False
    if cleaned_text.startswith('(') and cleaned_text.endswith(')'):
        cleaned_text = cleaned_text[1:-1]
        is_negative = True
    
    # Handle other currency symbols
    currency_symbols = ['$', '₹', '€', '£', '¥', 'Rs', 'Rs.', '₹.']
    for symbol in currency_symbols:
        cleaned_text = cleaned_text.replace(symbol, '')
    
    cleaned_text = cleaned_text.strip()
    
    # Try to convert to float
    try:
        value = float(cleaned_text)
        if is_negative:
            value = -value
        return value
    except ValueError:
        # If it's not convertible to float, return the original text
        return text

def format_result(financial_data, report_type, is_consolidated):
    """
    Format the extracted data into the standardized output structure.
    
    Args:
        financial_data: Dictionary containing extracted financial data
        report_type: Type of financial report
        is_consolidated: Boolean indicating if report is consolidated
        
    Returns:
        Dictionary in the standardized output format
    """
    result = {
        "Standalone_financial_results_for_all_months": {},
        "Balance_sheet": "Balance_sheet_are_not_present",
        "Cash_flow_statements": "Cash_flow_statements_are_not_present",
        "Statement_Consolidated_finanacial_results_for_all_months": {}
    }
    
    # Determine which section to populate
    if is_consolidated:
        target_section = "Statement_Consolidated_finanacial_results_for_all_months"
    else:
        target_section = "Standalone_financial_results_for_all_months"
    
    # Populate the appropriate section
    if report_type == "balance_sheet":
        result["Balance_sheet"] = financial_data
    elif report_type == "cash_flow":
        result["Cash_flow_statements"] = financial_data
    else:  # income_statement is default
        for period_type, periods in financial_data.items():
            if period_type not in result[target_section]:
                result[target_section][period_type] = {}
            
            for period_text, data in periods.items():
                result[target_section][period_type][period_text] = data
    
    # If the consolidated section is empty and not used, set it to not present
    if not result["Statement_Consolidated_finanacial_results_for_all_months"]:
        result["Statement_Consolidated_finanacial_results_for_all_months"] = "Statement_Consolidated_finanacial_results_for_all_months_are_not_present"
    
    # If the standalone section is empty and not used, set it to not present
    if not result["Standalone_financial_results_for_all_months"]:
        result["Standalone_financial_results_for_all_months"] = "Standalone_financial_results_for_all_months_are_not_present"
    
    return result

def create_empty_result():
    """
    Create an empty result structure when no data is found.
    
    Returns:
        Dictionary with empty structure
    """
    return {
        "Standalone_financial_results_for_all_months": {},
        "Balance_sheet": "Balance_sheet_are_not_present",
        "Cash_flow_statements": "Cash_flow_statements_are_not_present",
        "Statement_Consolidated_finanacial_results_for_all_months": "Statement_Consolidated_finanacial_results_for_all_months_are_not_present"
    }

def process_table_image(image_path, model, feature_extractor, output_path=None):
    """
    Main function to process table image and convert to JSON.

    Args:
    image_path (str): Path to input image
    output_path (str, optional): Path to save output JSON. If None, returns the parsed data without saving.

    Returns:
    dict: Parsed table data
    """
    try:

        # Preprocess image
        image = preprocess_image(image_path)

        # Detect table regions
        table_regions = detect_table_regions(image, model, feature_extractor)

        # Extract table from image
        table_image = extract_table_from_image(image, table_regions)

        # Perform OCR
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=r"./model", download_enabled=False)
        
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
