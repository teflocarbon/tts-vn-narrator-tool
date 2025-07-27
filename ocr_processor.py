"""
OCR processing module for the TTS Visual Novel Narrator Tool.
Handles text extraction with smart caching and optimization.
"""

import cv2
import numpy as np
import tempfile
import os
from typing import Optional
from difflib import SequenceMatcher
from ocrmac import ocrmac
from logger import setup_logger, log_ocr_status, log_similarity_check


class OCRProcessor:
    def __init__(self, debug: bool = False, similarity_threshold: float = 0.8):
        """
        Initialize the OCR processor.
        
        Args:
            debug: Enable debug mode with detailed logging
            similarity_threshold: Threshold for text similarity comparison
        """
        self.debug = debug
        self.similarity_threshold = similarity_threshold
        self.last_detected_text = ""
        self.logger = setup_logger('ocr', 'DEBUG' if debug else 'INFO')
        
        log_ocr_status("OCR Engine: Apple macOS OCR (via ocrmac)", "info")
        
        if debug:
            log_ocr_status("Debug mode enabled", "info")
    
    def extract_text_with_macos_ocr(self, image: np.ndarray) -> str:
        """
        Extract text from an image using Apple's macOS OCR.
        
        Args:
            image: Image to extract text from
            
        Returns:
            Extracted text as string
        """
        if image is None or image.size == 0:
            self.logger.debug("Empty or invalid image provided")
            return ""
        
        try:
            # Save image to temporary file for OCR processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Convert image format if needed
                if len(image.shape) == 3:
                    # Convert RGB to BGR for OpenCV
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image
                
                # Save image
                cv2.imwrite(temp_path, image_bgr)
                
                if self.debug:
                    self.logger.debug(f"Image saved to {temp_path}")
                
                # Run OCR
                annotations = ocrmac.OCR(temp_path).recognize()
                
                if self.debug:
                    log_ocr_status(f"Extracted {len(annotations)} annotations", "info")
                
                # Extract text from annotations
                # Annotations format: [(text, confidence, [x, y, width, height]), ...]
                text_parts = []
                for annotation in annotations:
                    if len(annotation) >= 1:
                        text = annotation[0].strip()
                        if text:
                            text_parts.append(text)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                
                # Combine text parts
                extracted_text = ' '.join(text_parts).strip()
                
                if extracted_text:
                    self.logger.debug(f"Extracted text: '{extracted_text[:50]}...'")
                    log_ocr_status(f"Successfully extracted {len(extracted_text)} characters", "success")
                else:
                    self.logger.debug("No text extracted")
                
                return extracted_text
                
        except Exception as e:
            log_ocr_status(f"OCR extraction failed: {e}", "error")
            self.logger.error(f"OCR error: {e}")
            return ""
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for similarity calculation
        similarity = SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
        
        self.logger.debug(f"Text similarity: {similarity:.3f}")
        return similarity
    
    def process_text_if_different(self, text: str) -> tuple[bool, str]:
        """
        Process text only if it's significantly different from the last detected text.
        
        Args:
            text: The text to process
            
        Returns:
            Tuple of (should_process, processed_text)
        """
        if not text.strip():
            return False, ""
        
        # Calculate similarity with last detected text
        similarity = self.calculate_text_similarity(text, self.last_detected_text)
        
        if similarity < self.similarity_threshold:
            # Text is significantly different - should process
            log_similarity_check(similarity, self.similarity_threshold, skipped=False)
            self.last_detected_text = text
            return True, text
        else:
            # Text is too similar - skip processing
            log_similarity_check(similarity, self.similarity_threshold, skipped=True)
            return False, text
    
    def reset_last_text(self):
        """Reset the last detected text (useful when starting fresh)."""
        self.last_detected_text = ""
        log_ocr_status("Reset last detected text", "info")
    
    def set_similarity_threshold(self, threshold: float):
        """
        Update the similarity threshold.
        
        Args:
            threshold: New similarity threshold (0.0 to 1.0)
        """
        old_threshold = self.similarity_threshold
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        log_ocr_status(f"Similarity threshold changed: {old_threshold:.2f} → {self.similarity_threshold:.2f}", "info")
