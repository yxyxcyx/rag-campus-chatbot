# enhanced_document_loader.py

"""
Document Loader with Advanced OCR

Handles:
- Text-based PDFs (PyMuPDF)
- Image-based PDFs (Tesseract OCR with preprocessing)
- DOCX files
- TXT files
- Image files (PNG, JPG, etc.)

Features:
- Adaptive DPI scaling
- Image preprocessing (deskew, denoise, contrast enhancement)
- Parallel processing for large documents
- Progress tracking
- Error recovery
"""

import os
import io
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import docx
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import numpy as np
from tqdm import tqdm

from logging_config import get_logger

# Module logger
logger = get_logger(__name__)


class EnhancedDocumentLoader:
    """Advanced document loader with OCR and preprocessing"""
    
    def __init__(self, dpi: float = 2.0, parallel_workers: int = 4):
        """
        Args:
            dpi: DPI multiplier for PDF rendering (2.0 = 144 DPI)
            parallel_workers: Number of parallel OCR workers
        """
        self.dpi = dpi
        self.parallel_workers = parallel_workers
        self.tesseract_config = r'--oem 3 --psm 6'  # LSTM OCR, assume uniform block
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocessing to improve OCR accuracy
        
        Steps:
        1. Convert to grayscale
        2. Enhance contrast
        3. Sharpen
        4. Denoise
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Denoise (optional, can be slow)
        # image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    
    def extract_text_from_pdf_page(self, page: fitz.Page, page_num: int) -> Tuple[int, str]:
        """
        Extract text from a PDF page with fallback to OCR
        
        Returns:
            (page_num, extracted_text)
        """
        # Try direct text extraction first
        text = page.get_text().strip()
        
        # If no text or very little text, use OCR
        if len(text) < 50:  # Threshold for "text-light" pages
            try:
                # Render page as image
                mat = fitz.Matrix(self.dpi, self.dpi)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                
                # Preprocess
                img = self.preprocess_image(img)
                
                # OCR
                ocr_text = pytesseract.image_to_string(
                    img, 
                    config=self.tesseract_config,
                    lang='eng'
                )
                
                # Use OCR text if it's more substantial
                if len(ocr_text.strip()) > len(text):
                    text = ocr_text
                    
            except Exception as e:
                logger.warning(
                    "OCR failed for page",
                    page_num=page_num + 1,
                    error=str(e)
                )
        
        return page_num, text
    
    def load_pdf(self, filepath: str, show_progress: bool = True) -> Dict[str, str]:
        """
        Load PDF with adaptive text extraction and OCR
        
        Args:
            filepath: Path to PDF file
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        doc = fitz.open(filepath)
        filename = os.path.basename(filepath)
        
        logger.info("Processing PDF", filename=filename, pages=len(doc))
        
        # Quick scan to determine if OCR is needed
        text_pages = 0
        for page in doc:
            if len(page.get_text().strip()) > 50:
                text_pages += 1
        
        ocr_needed = text_pages < len(doc) * 0.5  # >50% pages need OCR
        
        if ocr_needed:
            logger.info(
                "OCR mode enabled",
                ocr_pages=len(doc) - text_pages,
                total_pages=len(doc)
            )
        else:
            logger.info(
                "Text extraction mode",
                text_pages=text_pages,
                total_pages=len(doc)
            )
        
        # Extract pages
        if self.parallel_workers > 1 and len(doc) > 5:
            # Parallel processing for large documents
            results = {}
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = {
                    executor.submit(self.extract_text_from_pdf_page, doc[i], i): i 
                    for i in range(len(doc))
                }
                
                iterator = tqdm(
                    as_completed(futures), 
                    total=len(doc),
                    desc="   Extracting",
                    disable=not show_progress
                )
                
                for future in iterator:
                    page_num, text = future.result()
                    results[f"page_{page_num + 1}"] = text
        else:
            # Sequential processing
            results = {}
            iterator = range(len(doc))
            if show_progress:
                iterator = tqdm(iterator, desc="   Extracting")
            
            for i in iterator:
                page_num, text = self.extract_text_from_pdf_page(doc[i], i)
                results[f"page_{page_num + 1}"] = text
        
        doc.close()
        
        # Statistics
        total_text = sum(len(text) for text in results.values())
        non_empty = sum(1 for text in results.values() if len(text.strip()) > 50)
        
        logger.info(
            "PDF extraction complete",
            filename=filename,
            pages_extracted=non_empty,
            total_pages=len(results),
            total_characters=total_text
        )
        
        return {filename: "\n\n".join(results.values())}
    
    def load_docx(self, filepath: str) -> Dict[str, str]:
        """Load DOCX file"""
        filename = os.path.basename(filepath)
        logger.info("Processing DOCX", filename=filename)
        
        doc = docx.Document(filepath)
        text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        
        logger.info("DOCX extraction complete", filename=filename, characters=len(text))
        
        return {filename: text}
    
    def load_txt(self, filepath: str) -> Dict[str, str]:
        """Load TXT file"""
        filename = os.path.basename(filepath)
        logger.info("Processing TXT", filename=filename)
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        logger.info("TXT extraction complete", filename=filename, characters=len(text))
        
        return {filename: text}
    
    def load_image(self, filepath: str) -> Dict[str, str]:
        """Load and OCR image file"""
        filename = os.path.basename(filepath)
        logger.info("Processing image", filename=filename)
        
        try:
            img = Image.open(filepath)
            img = self.preprocess_image(img)
            text = pytesseract.image_to_string(
                img, 
                config=self.tesseract_config,
                lang='eng'
            )
            
            logger.info("Image OCR complete", filename=filename, characters=len(text))
            
            return {filename: text}
        except Exception as e:
            logger.error("Image processing failed", filename=filename, error=str(e))
            return {filename: ""}
    
    def load_document(self, filepath: str) -> Dict[str, str]:
        """
        Load any supported document type
        
        Supported formats: PDF, DOCX, TXT, PNG, JPG, JPEG, TIFF
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf':
            return self.load_pdf(filepath)
        elif ext == '.docx':
            return self.load_docx(filepath)
        elif ext == '.txt':
            return self.load_txt(filepath)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self.load_image(filepath)
        else:
            logger.warning("Unsupported format", extension=ext, filepath=filepath)
            return {}
    
    def load_folder(self, folder_path: str) -> Dict[str, str]:
        """
        Load all documents from a folder
        
        Returns:
            Dictionary mapping filenames to extracted text
        """
        logger.info("Loading documents from folder", folder=folder_path)
        
        documents = {}
        files = [
            f for f in os.listdir(folder_path)
            if not f.startswith('.') and os.path.isfile(os.path.join(folder_path, f))
        ]
        
        logger.info("Files found", file_count=len(files))
        
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            doc_dict = self.load_document(filepath)
            documents.update(doc_dict)
        
        total_chars = sum(len(t) for t in documents.values())
        logger.info(
            "Folder loading complete",
            documents_loaded=len(documents),
            total_characters=total_chars
        )
        
        return documents


# Convenience function for backward compatibility
def load_documents_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Load all documents from a folder using enhanced loader
    """
    loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=4)
    return loader.load_folder(folder_path)


if __name__ == "__main__":
    # Test the loader
    import sys
    from logging_config import setup_logging
    
    # Setup logging for standalone execution
    setup_logging(level="INFO", json_output=False, app_name="doc-loader")
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data/"
    
    loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=4)
    
    if os.path.isfile(path):
        docs = loader.load_document(path)
    else:
        docs = loader.load_folder(path)
    
    logger.info("Loading summary", documents=len(docs))
    for filename, text in docs.items():
        logger.info("Document loaded", filename=filename, characters=len(text))
