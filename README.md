# Table Parser

A powerful document table extraction tool built with Streamlit and PyTorch.

[![GitHub](https://img.shields.io/badge/GitHub-Table--Parser-blue)](https://github.com/amanrjgit/Table-parser.git)

## Overview

Table Parser is a web application that extracts tables from PDF documents and images. It uses the Microsoft Table Transformer (TAFT) model for table detection and OCR processing to convert visual tables into structured data.

## Features

- PDF document uploading and processing
- Direct image upload support (PNG, JPG, JPEG)
- Automatic table detection using state-of-the-art AI models
- Interactive image selection from multi-page PDFs
- Extraction of tabular data into structured JSON format
- Simple and intuitive user interface

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch
- CUDA-capable GPU (recommended but not required)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/amanrjgit/Table-parser.git
   cd Table-parser
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Launch the application in your web browser.
2. Upload a PDF document or image containing tables.
3. For PDFs, select the images containing tables from the displayed thumbnails.
4. Click the "Extract Tables" button to process the selected images.
5. View the extracted data in JSON format.
6. Download the extracted data using the "Download JSON" button.

## Important Notes

- **Image Orientation**: For best results, upload images with proper orientation. The model may have difficulty detecting tables in rotated or skewed images.
- The first run may take some time as it downloads the pre-trained models.
- Processing large PDFs or complex tables may require additional processing time.
- GPU acceleration significantly improves performance if available.

## How It Works

The application uses a two-stage process:
1. **Table Detection**: Microsoft's Table Transformer model identifies table regions within the document.
2. **OCR Processing**: The detected table regions are processed to extract structured tabular data.

## Troubleshooting

- If you encounter memory issues, try processing fewer images at once.
- For optimal performance, ensure your images are clear and tables have distinct borders.
- If the application fails to detect tables, try converting your document to images with higher contrast.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Microsoft Table Transformer](https://github.com/microsoft/table-transformer) for the table detection model
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing
- [Streamlit](https://streamlit.io/) for the web application framework
