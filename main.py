#!/usr/bin/env python3
"""
TTS Visual Novel Narrator Tool

This application captures a selected region of the screen, monitors for changes,
and uses OCR to extract text when the content stops scrolling.
"""

import cv2
import numpy as np
import pytesseract
import mss
import time
import threading
from typing import Tuple, Optional
import argparse
from difflib import SequenceMatcher

# Screen capture imports
try:
    from PIL import Image, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Install with: pip install Pillow")


class ScreenRegionMonitor:
    def __init__(self, check_interval: float = 1.0, debug: bool = False, capture_method: str = "auto"):
        """
        Initialize the screen region monitor.
        
        Args:
            check_interval: Time in seconds between snapshots (default: 1.0)
            debug: Enable debug mode with image saving and previews
            capture_method: Screen capture method ('mss', 'pil', 'auto')
        """
        self.check_interval = check_interval
        self.monitoring = False
        self.selected_region = None
        self.previous_image = None
        self.monitor_thread = None
        self.sct = mss.mss()
        self.debug = debug
        self.debug_counter = 0
        self.capture_method = capture_method
        
        # Text detection state management
        self.last_detected_text = ""
        self.waiting_for_change = False
        self.text_similarity_threshold = 0.8  # 80% similarity threshold
        
        print("OCR Engine: Tesseract (simplified)")
        if debug:
            print("DEBUG MODE ENABLED - Images will be saved to debug_captures/")
            import os
            os.makedirs("debug_captures", exist_ok=True)
        
    def select_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Allow user to select a region of the screen by clicking and dragging.
        Uses a more efficient approach with a transparent overlay.
        
        Returns:
            Tuple of (x, y, width, height) or None if cancelled
        """
        print("Select the region to monitor...")
        print("Instructions:")
        print("- Click and drag to select the region")
        print("- Press ENTER or SPACE to confirm")
        print("- Press ESC or 'q' to cancel")
        
        # Get screen dimensions
        monitor = self.sct.monitors[0]
        screen_width = monitor['width']
        screen_height = monitor['height']
        
        # Create a smaller preview for better performance
        scale_factor = 0.5  # Scale down for better performance
        preview_width = int(screen_width * scale_factor)
        preview_height = int(screen_height * scale_factor)
        
        # Capture and resize screenshot
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_resized = cv2.resize(img, (preview_width, preview_height))
        
        # Variables for selection
        selecting = False
        start_point = None
        end_point = None
        current_img = img_resized.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selecting, start_point, end_point, current_img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_point = (x, y)
                end_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                end_point = (x, y)
                # Only redraw when actively selecting
                current_img = img_resized.copy()
                if start_point and end_point:
                    cv2.rectangle(current_img, start_point, end_point, (0, 255, 0), 2)
                    # Add selection info
                    width = abs(end_point[0] - start_point[0])
                    height = abs(end_point[1] - start_point[1])
                    cv2.putText(current_img, f"Size: {int(width/scale_factor)}x{int(height/scale_factor)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                if start_point and end_point:
                    cv2.rectangle(current_img, start_point, end_point, (0, 255, 0), 2)
                    width = abs(end_point[0] - start_point[0])
                    height = abs(end_point[1] - start_point[1])
                    cv2.putText(current_img, f"Size: {int(width/scale_factor)}x{int(height/scale_factor)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(current_img, "Press SPACE/ENTER to confirm, ESC/Q to cancel", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Create window
        window_name = "Select Region (Scaled View)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, preview_width, preview_height)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Add initial instructions
        cv2.putText(current_img, "Click and drag to select region", 
                   (10, preview_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        while True:
            cv2.imshow(window_name, current_img)
            key = cv2.waitKey(30) & 0xFF  # Increased wait time for better performance
            
            if key == ord('q') or key == 27:  # ESC key
                cv2.destroyAllWindows()
                return None
                
            elif key == ord(' ') or key == 13:  # SPACE or ENTER
                if start_point and end_point:
                    cv2.destroyAllWindows()
                    # Calculate region coordinates (scale back to original size)
                    x1, y1 = start_point
                    x2, y2 = end_point
                    x = int(min(x1, x2) / scale_factor)
                    y = int(min(y1, y2) / scale_factor)
                    width = int(abs(x2 - x1) / scale_factor)
                    height = int(abs(y2 - y1) / scale_factor)
                    
                    # Ensure minimum size
                    if width < 50 or height < 50:
                        print("Selected region too small. Please select a larger area.")
                        return self.select_region()  # Retry
                    
                    return (x, y, width, height)
        
        cv2.destroyAllWindows()
        return None
    
    def capture_region(self) -> np.ndarray:
        """
        Capture the selected region of the screen using the configured method.
        
        Returns:
            Numpy array of the captured image
        """
        if not self.selected_region:
            raise ValueError("No region selected")
            
        x, y, width, height = self.selected_region
        
        # Try different capture methods
        img = None
        method_used = ""
        
        if self.capture_method in ["mss", "auto"]:
            try:
                img = self._capture_with_mss(x, y, width, height)
                method_used = "MSS"
            except Exception as e:
                if self.debug:
                    print(f"DEBUG: MSS capture failed: {e}")
                if self.capture_method == "mss":
                    raise
        
        if img is None and self.capture_method in ["pil", "auto"] and PIL_AVAILABLE:
            try:
                img = self._capture_with_pil(x, y, width, height)
                method_used = "PIL"
            except Exception as e:
                if self.debug:
                    print(f"DEBUG: PIL capture failed: {e}")
                if self.capture_method == "pil":
                    raise
        
        if img is None:
            raise RuntimeError("All screen capture methods failed")
        
        if self.debug:
            self.debug_counter += 1
            debug_filename = f"debug_captures/capture_{self.debug_counter:04d}.png"
            try:
                cv2.imwrite(debug_filename, img)
                print(f"DEBUG: Saved captured image to {debug_filename} (size: {img.shape}) using {method_used}")
                
                # Show a preview window (with error handling)
                try:
                    preview = cv2.resize(img, (min(800, img.shape[1]), min(600, img.shape[0])))
                    cv2.imshow("Debug: Captured Region", preview)
                    cv2.waitKey(1)  # Non-blocking
                except Exception as e:
                    print(f"DEBUG: Preview window failed: {e}")
            except Exception as e:
                print(f"DEBUG: Failed to save debug image: {e}")
        
        return img
    
    def _capture_with_mss(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Capture screen region using MSS library.
        """
        monitor = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }
        
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
    def _capture_with_pil(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Capture screen region using PIL ImageGrab.
        """
        # PIL uses (left, top, right, bottom) format
        bbox = (x, y, x + width, y + height)
        screenshot = ImageGrab.grab(bbox)
        
        # Convert PIL Image to numpy array
        img = np.array(screenshot)
        
        # Convert RGB to BGR for OpenCV compatibility
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return img
    
    def images_are_different(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Compare two images to determine if they are different.
        
        Args:
            img1: First image
            img2: Second image
            threshold: Difference threshold (0.0 to 1.0)
            
        Returns:
            True if images are different, False if they are the same
        """
        try:
            if img1 is None or img2 is None:
                return True
                
            if img1.shape != img2.shape:
                if self.debug:
                    print(f"DEBUG: Image shapes differ: {img1.shape} vs {img2.shape}")
                return True
            
            # Validate image data
            if img1.size == 0 or img2.size == 0:
                if self.debug:
                    print("DEBUG: Empty image detected")
                return True
            
            # Convert to grayscale for comparison (with error handling)
            try:
                if len(img1.shape) == 3:
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                else:
                    gray1 = img1.copy()
                    
                if len(img2.shape) == 3:
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                else:
                    gray2 = img2.copy()
            except Exception as e:
                if self.debug:
                    print(f"DEBUG: Color conversion failed: {e}")
                # Fallback to simple numpy comparison
                return not np.array_equal(img1, img2)
            
            # Calculate the difference (with error handling)
            try:
                diff = cv2.absdiff(gray1, gray2)
            except Exception as e:
                if self.debug:
                    print(f"DEBUG: cv2.absdiff failed: {e}")
                # Fallback to numpy difference
                diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
            
            # Calculate the percentage of different pixels
            total_pixels = diff.size
            if total_pixels == 0:
                return False
                
            different_pixels = np.count_nonzero(diff > 10)  # Small threshold for noise
            difference_ratio = different_pixels / total_pixels
            
            if self.debug:
                print(f"DEBUG: Image difference ratio: {difference_ratio:.4f} (threshold: {threshold})")
            
            return difference_ratio > threshold
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Image comparison failed with exception: {e}")
            # If comparison fails, assume images are different to be safe
            return True
    

    
    def extract_text_with_ocr(self, image: np.ndarray) -> str:
        """
        Extract text from an image using Tesseract OCR with minimal preprocessing.
        
        Args:
            image: Image to extract text from
            
        Returns:
            Extracted text as string
        """
        height, width = image.shape[:2]
        print(f"Processing image: {width}x{height} with Tesseract (simplified)")
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Debug: Save the original image being processed
            if self.debug:
                debug_filename = f"debug_captures/tesseract_original_{self.debug_counter:04d}.png"
                cv2.imwrite(debug_filename, gray)
                print(f"DEBUG: Saved original OCR input to {debug_filename}")
            
            # Use Tesseract with simple configuration
            text = pytesseract.image_to_string(gray, config='--psm 6')
            text = text.strip()
            
            # Clean up the text
            if text:
                text = ' '.join(text.split())  # Normalize whitespace
                print(f"Tesseract result: '{text}'")
            else:
                print("No text detected")
            
            return text
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
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
            
        # Normalize texts for comparison
        text1_norm = ' '.join(text1.lower().split())
        text2_norm = ' '.join(text2.lower().split())
        
        return SequenceMatcher(None, text1_norm, text2_norm).ratio()
    
    def on_text_detected(self, text: str):
        """
        Handle detected text. This is where TTS integration will be added later.
        Only outputs text if it's significantly different from the last detected text.
        
        Args:
            text: The detected text
        """
        if text and len(text.strip()) > 0:
            # Check if this text is significantly different from the last one
            similarity = self.calculate_text_similarity(text, self.last_detected_text)
            
            if similarity < self.text_similarity_threshold:
                print(f"\n{'='*50}")
                print("NEW TEXT DETECTED:")
                print(f"{'='*50}")
                print(text)
                print(f"{'='*50}")
                if self.last_detected_text:
                    print(f"Similarity to previous: {similarity:.2%}")
                print()
                
                # Update state
                self.last_detected_text = text
                self.waiting_for_change = True
                
                # TODO: Add TTS integration here
                # tts_engine.speak(text)
            else:
                print(f"Similar text detected (similarity: {similarity:.2%}) - skipping output")
        else:
            print("No meaningful text detected in the region.")
    
    def monitor_region(self):
        """
        Monitor the selected region for changes and extract text when stable.
        Implements smart state management to avoid repeated OCR of the same text.
        """
        print(f"Starting monitoring with {self.check_interval}s interval...")
        print("Press Ctrl+C to stop monitoring.")
        print(f"Text similarity threshold: {self.text_similarity_threshold:.0%}")
        print()
        
        while self.monitoring:
            try:
                # Capture current image
                current_image = self.capture_region()
                
                if self.previous_image is not None:
                    # Check if there's a change in the image
                    has_change = self.images_are_different(current_image, self.previous_image)
                    
                    if self.waiting_for_change:
                        # We detected text before and are waiting for a change
                        if has_change:
                            print("Change detected after text detection - checking new content...")
                            # Extract text to see if it's actually different
                            text = self.extract_text_with_ocr(current_image)
                            self.on_text_detected(text)
                            # Reset waiting state - on_text_detected will set it again if needed
                            self.waiting_for_change = False
                        else:
                            print("Waiting for change (text already detected)...")
                    else:
                        # Normal monitoring mode
                        if not has_change:
                            print("No change detected - extracting text...")
                            text = self.extract_text_with_ocr(current_image)
                            self.on_text_detected(text)
                        else:
                            print("Change detected - waiting for stability...")
                            # Reset waiting state since we're seeing changes
                            self.waiting_for_change = False
                else:
                    # First image capture
                    print("Initial capture - waiting for next frame...")
                
                # Update previous image
                self.previous_image = current_image.copy()
                
                # Wait for the specified interval
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(1)  # Brief pause before retrying
    
    def start_monitoring(self):
        """Start monitoring in a separate thread."""
        if self.monitoring:
            print("Already monitoring!")
            return
            
        if not self.selected_region:
            print("No region selected!")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_region, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()


def main():
    parser = argparse.ArgumentParser(description="TTS Visual Novel Narrator Tool")
    parser.add_argument(
        "--interval", 
        type=float, 
        default=1.0,
        help="Check interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--capture-method",
        choices=["mss", "pil", "auto"],
        default="auto",
        help="Screen capture method: mss, pil, or auto (default: auto)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with image saving and detailed output"
    )
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = ScreenRegionMonitor(
        check_interval=args.interval, 
        debug=args.debug,
        capture_method=args.capture_method
    )
    
    # Select region
    region = monitor.select_region()
    if not region:
        print("No region selected. Exiting.")
        return
    
    monitor.selected_region = region
    print(f"Selected region: {region}")
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Keep the main thread alive
        while monitor.monitoring:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        monitor.stop_monitoring()
        print("Goodbye!")


if __name__ == "__main__":
    main()
