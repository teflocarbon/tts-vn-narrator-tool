"""
OCR processing module for the TTS Visual Novel Narrator Tool.
Handles text extraction with smart caching and optimization.
"""

import numpy as np
import tempfile
import os
from typing import Optional
from difflib import SequenceMatcher
from PIL import Image, ImageEnhance
from ocrmac import ocrmac
from logger import setup_logger, log_ocr_status, log_similarity_check


class OCRProcessor:
    def __init__(self, debug: bool = False, similarity_threshold: float = 0.8, postprocess_images: bool = False):
        """
        Initialize the OCR processor.
        
        Args:
            debug: Enable debug mode with detailed logging
            similarity_threshold: Threshold for text similarity comparison
            postprocess_images: Enable image postprocessing for better OCR accuracy
        """
        self.debug = debug
        self.similarity_threshold = similarity_threshold
        self.postprocess_images = postprocess_images
        self.last_detected_text = ""
        self.logger = setup_logger('ocr', 'DEBUG' if debug else 'INFO')
        
        log_ocr_status("OCR Engine: Apple macOS OCR (via ocrmac)", "info")
        
        if debug:
            log_ocr_status("Debug mode enabled", "info")
        
        if postprocess_images:
            log_ocr_status("Image postprocessing enabled (desaturation + high contrast)", "info")
    
    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Postprocess image to improve OCR accuracy by removing saturation and increasing contrast.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Postprocessed image as numpy array
        """
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                # RGB image
                pil_image = Image.fromarray(image.astype('uint8'))
            else:
                # Grayscale image
                pil_image = Image.fromarray(image.astype('uint8'), mode='L')
            
            # Remove saturation (convert to grayscale while preserving luminance)
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            
            # Significantly increase contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.5)  # Increase contrast by 2.5x
            
            # Convert back to numpy array
            processed_array = np.array(pil_image)
            
            if self.debug:
                self.logger.debug("Applied image postprocessing: desaturation + high contrast")
            
            return processed_array
            
        except Exception as e:
            self.logger.warning(f"Image postprocessing failed: {e}, using original image")
            return image
    
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
            # Apply postprocessing if enabled
            processed_image = self._postprocess_image(image) if self.postprocess_images else image
            
            # Save image to temporary file for OCR processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Convert numpy array to PIL Image and save
                if len(processed_image.shape) == 3:
                    # RGB image
                    pil_image = Image.fromarray(processed_image.astype('uint8'))
                else:
                    # Grayscale image
                    pil_image = Image.fromarray(processed_image.astype('uint8'), mode='L')
                
                # Save image
                pil_image.save(temp_path, 'PNG')
                
                if self.debug:
                    self.logger.debug(f"Image saved to {temp_path}")
                
                # Run OCR with error handling
                try:
                    annotations = ocrmac.OCR(temp_path).recognize()
                except Exception as ocr_error:
                    log_ocr_status(f"OCR processing failed: {ocr_error}", "error")
                    return ""
                
                # Check if OCR returned valid results
                if annotations is None:
                    if self.debug:
                        log_ocr_status("OCR returned None - no annotations found", "warning")
                    return ""
                
                if self.debug:
                    log_ocr_status(f"Extracted {len(annotations)} annotations", "info")
                
                # Extract text from annotations
                # Annotations format: [(text, confidence, [x, y, width, height]), ...]
                text_parts = []
                try:
                    for annotation in annotations:
                        if annotation is None:
                            continue
                        if len(annotation) >= 1 and annotation[0] is not None:
                            text = str(annotation[0]).strip()
                            if text:
                                text_parts.append(text)
                except (TypeError, IndexError, AttributeError) as parse_error:
                    log_ocr_status(f"Error parsing annotations: {parse_error}", "error")
                    if self.debug:
                        self.logger.debug(f"Problematic annotations: {annotations}")
                
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
