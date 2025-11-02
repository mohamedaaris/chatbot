import io
import os
from typing import Optional
import urllib3

import pdfplumber
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
try:
    # python-docx
    from docx import Document  # type: ignore
except Exception:
    Document = None  # gracefully handle environments without python-docx

# Disable SSL warnings for sites with certificate issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber; fallback to OCR for image-only PDFs.

    The function first tries native text extraction. If very little text is found,
    it rasterizes pages and runs OCR via Tesseract to capture scanned content.
    """
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            texts.append(txt)

    combined = "\n".join(t.strip() for t in texts if t)
    if len(combined) >= 200:
        return combined

    # Fallback to OCR per-page if text is scarce (likely a scanned PDF)
    with pdfplumber.open(path) as pdf:
        ocr_texts = []
        for page in pdf.pages:
            try:
                img = page.to_image(resolution=300)
                pil_img = Image.open(io.BytesIO(img.original)) if isinstance(img.original, (bytes, bytearray)) else img.image
                ocr_text = pytesseract.image_to_string(pil_img)
                if ocr_text:
                    ocr_texts.append(ocr_text)
            except Exception:
                # If any single page fails OCR, continue with others
                continue
    return "\n".join(t.strip() for t in ocr_texts if t)


def extract_text_from_image(path: str) -> str:
    """Extract text from an image using Tesseract OCR."""
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    return text.strip()


def extract_text_from_docx(path: str) -> str:
    """Extract text from a DOCX using python-docx.

    Returns a single string with paragraph text joined by newlines. If python-docx
    is not available, raises a helpful error so the caller can report it.
    """
    if Document is None:
        raise RuntimeError("python-docx is not installed. Please install it with 'pip install python-docx'.")

    try:
        doc = Document(path)
        parts = []
        for para in doc.paragraphs:
            text = (para.text or "").strip()
            if text:
                parts.append(text)
        # Optionally capture simple table cell text
        for tbl in getattr(doc, 'tables', []):
            for row in tbl.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if row_text:
                    parts.append(" \t ".join(row_text))
        return "\n".join(parts)
    except Exception as e:
        raise RuntimeError(f"Failed to extract DOCX text: {e}")


def clean_web_text(text: str) -> str:
    """Simple cleanup for web text blocks."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def extract_text_from_url(url: str, timeout: int = 20) -> str:
    """Fetch a webpage and extract clean readable text using BeautifulSoup.

    Enhanced extraction with better content filtering and cleaning.
    Only online operation in the app; everything else works offline.
    """
    try:
        # Add SSL verification bypass for problematic certificates
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove unwanted elements more comprehensively
        unwanted_tags = [
            "script", "style", "nav", "footer", "noscript", "header", "aside",
            "advertisement", "ads", "sidebar", "menu", "navigation", "social",
            "comment", "comments", "share", "related", "recommended"
        ]
        for tag in soup(unwanted_tags):
            tag.decompose()

        # Remove elements with unwanted classes/ids
        unwanted_classes = ["ad", "ads", "advertisement", "sidebar", "menu", "nav", "social", "share"]
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=lambda x: x and any(cls in x.lower() for cls in class_name.split())):
                element.decompose()

        # Prefer main content areas
        content_selectors = [
            "main", "article", "[role='main']", ".content", ".post", ".article",
            ".entry-content", ".post-content", ".article-content", "#content"
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find("body") or soup

        # Extract text with better formatting
        text = main_content.get_text(separator="\n", strip=True)
        cleaned = clean_web_text(text)
        
        # Ensure we have meaningful content
        if len(cleaned) < 100:
            raise ValueError("Insufficient text content found")
            
        return cleaned
    except Exception as e:
        raise Exception(f"Failed to extract text from URL: {str(e)}")


