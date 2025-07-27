#!/usr/bin/env python3
"""
TTS Visual Novel Narrator Tool - macOS Only

This application captures a selected window on macOS, monitors for changes,
and uses Apple's Live Text OCR to extract text when the content stops scrolling.
"""

import cv2
import numpy as np
import time
import threading
import tempfile
import os
from typing import Optional
import argparse
from difflib import SequenceMatcher

# macOS-specific imports
from window_capture import WindowCapture
from ocrmac import ocrmac


class WindowMonitor:
    def __init__(self, check_interval: float = 1.0, debug: bool = False):
        """
        Initialize the macOS window monitor.
        
        Args:
            check_interval: Time in seconds between snapshots (default: 1.0)
            debug: Enable debug mode with image saving and detailed output
        """
        self.check_interval = check_interval
        self.monitoring = False
        self.previous_image = None
        self.monitor_thread = None
        self.debug = debug
        self.debug_counter = 0
        
        # Window capture setup
        self.window_capture = WindowCapture(debug=debug)
        
        # Text detection state management
        self.last_detected_text = ""
        self.waiting_for_change = False
        self.text_similarity_threshold = 0.8  # 80% similarity threshold
        
        print("OCR Engine: Apple Live Text (via ocrmac)")
        if debug:
            print("DEBUG MODE ENABLED - Images will be saved to debug_captures/")
            os.makedirs("debug_captures", exist_ok=True)
    
    def select_window(self) -> bool:
        """
        Allow user to select a window to monitor.
        
        Returns:
            True if a window was selected, False otherwise
        """
        return self.window_capture.select_window()
    
    def select_window_region(self) -> Optional[tuple[int, int, int, int]]:
        """
        Allow user to select a region within the captured window.
        
        Returns:
            Tuple of (x, y, width, height) relative to the window or None if cancelled
        """
        if not self.window_capture or not self.window_capture.target_window:
            print("No window selected for region selection!")
            return None
        
        print("\nSelect the region within the window to monitor...")
        print("Instructions:")
        print("- Click and drag to select the region")
        print("- Press ENTER or SPACE to confirm")
        print("- Press ESC or 'q' to cancel")
        print("- Press 'f' to use the full window (no region)")
        
        try:
            # Capture the current window
            window_img = self.window_capture.capture_current_window()
            
            # Convert to BGR for OpenCV display
            if len(window_img.shape) == 3 and window_img.shape[2] == 3:
                display_img = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
            else:
                display_img = window_img.copy()
            
            # Scale down for display if window is too large
            window_height, window_width = display_img.shape[:2]
            max_display_width = 1200
            max_display_height = 800
            
            scale_factor = 1.0
            if window_width > max_display_width or window_height > max_display_height:
                scale_x = max_display_width / window_width
                scale_y = max_display_height / window_height
                scale_factor = min(scale_x, scale_y)
                
                display_width = int(window_width * scale_factor)
                display_height = int(window_height * scale_factor)
                display_img = cv2.resize(display_img, (display_width, display_height))
            
            # Variables for selection
            selecting = False
            start_point = None
            end_point = None
            current_img = display_img.copy()
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal selecting, start_point, end_point, current_img
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    selecting = True
                    start_point = (x, y)
                    end_point = (x, y)
                    
                elif event == cv2.EVENT_MOUSEMOVE and selecting:
                    end_point = (x, y)
                    current_img = display_img.copy()
                    if start_point and end_point:
                        cv2.rectangle(current_img, start_point, end_point, (0, 255, 0), 2)
                        # Add selection info
                        width = abs(end_point[0] - start_point[0])
                        height = abs(end_point[1] - start_point[1])
                        cv2.putText(current_img, f"Selection: {width}x{height}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                elif event == cv2.EVENT_LBUTTONUP:
                    selecting = False
            
            cv2.namedWindow("Select Window Region", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Select Window Region", mouse_callback)
            
            while True:
                cv2.imshow("Select Window Region", current_img)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return None
                elif key == ord('f'):  # Full window
                    cv2.destroyAllWindows()
                    return None  # No region means full window
                elif key == 13 or key == 32:  # ENTER or SPACE
                    if start_point and end_point:
                        # Convert display coordinates back to actual window coordinates
                        actual_x1 = int(min(start_point[0], end_point[0]) / scale_factor)
                        actual_y1 = int(min(start_point[1], end_point[1]) / scale_factor)
                        actual_x2 = int(max(start_point[0], end_point[0]) / scale_factor)
                        actual_y2 = int(max(start_point[1], end_point[1]) / scale_factor)
                        
                        actual_width = actual_x2 - actual_x1
                        actual_height = actual_y2 - actual_y1
                        
                        if actual_width > 10 and actual_height > 10:  # Minimum size check
                            cv2.destroyAllWindows()
                            return (actual_x1, actual_y1, actual_width, actual_height)
                        else:
                            print("Selection too small. Please select a larger region.")
                    else:
                        print("Please make a selection first.")
                        
        except Exception as e:
            print(f"Error during window region selection: {e}")
            cv2.destroyAllWindows()
            return None
        
        cv2.destroyAllWindows()
        return None
    
    def capture_current_window(self) -> np.ndarray:
        """
        Capture the currently selected window.
        
        Returns:
            Numpy array of the captured window
        """
        return self.window_capture.capture_current_window()
    
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
        if img1 is None or img2 is None:
            return True
        
        if img1.shape != img2.shape:
            return True
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)
        
        # Convert to grayscale if needed
        if len(diff.shape) == 3:
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Calculate the percentage of different pixels
        non_zero_count = cv2.countNonZero(diff)
        total_pixels = diff.shape[0] * diff.shape[1]
        difference_ratio = non_zero_count / total_pixels
        
        if self.debug:
            print(f"Image difference ratio: {difference_ratio:.4f} (threshold: {threshold})")
        
        return difference_ratio > threshold
    
    def extract_text_with_live_text(self, image: np.ndarray) -> str:
        """
        Extract text from an image using Apple's Live Text OCR.
        
        Args:
            image: Image to extract text from
            
        Returns:
            Extracted text as string
        """
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                # Save the image
                cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            
            # Use ocrmac with Live Text - this returns a list directly
            annotations = ocrmac.livetext_from_image(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Extract text from annotations
            # Annotations format: [(text, confidence, [x, y, width, height]), ...]
            text_parts = []
            for annotation in annotations:
                if len(annotation) >= 1:
                    text = annotation[0].strip()
                    if text:
                        text_parts.append(text)
            
            # Join all text parts with spaces
            extracted_text = ' '.join(text_parts).strip()
            
            if self.debug:
                print(f"Live Text extracted {len(annotations)} annotations")
                print(f"Extracted text: '{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}'")
            
            return extracted_text
            
        except Exception as e:
            if self.debug:
                print(f"Error during Live Text OCR: {e}")
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
        matcher = SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
    
    def on_text_detected(self, text: str):
        """
        Handle detected text. This is where TTS integration will be added later.
        Only outputs text if it's significantly different from the last detected text.
        
        Args:
            text: The detected text
        """
        if not text or not text.strip():
            return
        
        # Check similarity with last detected text
        similarity = self.calculate_text_similarity(text, self.last_detected_text)
        
        if similarity < self.text_similarity_threshold:
            print(f"\n{'='*50}")
            print("NEW TEXT DETECTED:")
            print(f"{'='*50}")
            print(text)
            print(f"{'='*50}")
            
            # Update last detected text
            self.last_detected_text = text
            
            # TODO: Uncomment the line below when ready to add TTS
            # tts_engine.speak(text)
        else:
            if self.debug:
                print(f"Text similarity too high ({similarity:.2f}), skipping output")
    
    def monitor_window(self):
        """
        Monitor the selected window for changes and extract text when stable.
        Implements smart state management to avoid repeated OCR of the same text.
        """
        print("Starting window monitoring...")
        print("Press Ctrl+C to stop monitoring")
        
        while self.monitoring:
            try:
                # Capture current window
                current_image = self.capture_current_window()
                
                if self.debug and self.debug_counter % 10 == 0:  # Save every 10th frame in debug mode
                    debug_path = f"debug_captures/frame_{self.debug_counter:04d}.png"
                    cv2.imwrite(debug_path, current_image)
                    print(f"Debug: Saved frame to {debug_path}")
                
                self.debug_counter += 1
                
                # Compare with previous image
                if self.previous_image is not None:
                    if self.images_are_different(current_image, self.previous_image):
                        if self.waiting_for_change:
                            # We were waiting for change and got it - run OCR to see if text changed
                            text = self.extract_text_with_live_text(current_image)
                            if text:
                                self.on_text_detected(text)
                                # Now wait for the next change
                                self.waiting_for_change = True
                            else:
                                # No text found, keep waiting for changes
                                self.waiting_for_change = False
                        else:
                            print("Change detected - waiting for stability...")
                            # Reset waiting state since we're seeing changes
                            self.waiting_for_change = False
                    else:
                        # No change detected
                        if not self.waiting_for_change:
                            # Content has stabilized - run OCR
                            text = self.extract_text_with_live_text(current_image)
                            if text:
                                self.on_text_detected(text)
                                # Set flag to wait for next change
                                self.waiting_for_change = True
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
            
        # Check if we have a window selected
        if not self.window_capture or not self.window_capture.target_window:
            print("No window selected!")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_window, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()


def main():
    parser = argparse.ArgumentParser(description="TTS Visual Novel Narrator Tool - macOS Only")
    parser.add_argument(
        "--interval", 
        type=float, 
        default=1.0,
        help="Check interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with image saving and detailed output"
    )
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = WindowMonitor(
        check_interval=args.interval, 
        debug=args.debug
    )
    
    # Select window
    if not monitor.select_window():
        print("No window selected. Exiting.")
        return
    
    print("Window selected successfully.")
    
    # Optionally select a region within the window
    print("\nDo you want to select a specific region within the window?")
    choice = input("Press 'y' for region selection, or any other key to use the full window: ").strip().lower()
    
    if choice == 'y':
        window_region = monitor.select_window_region()
        if window_region:
            monitor.window_capture.set_region(window_region)
            print(f"Selected window region: {window_region}")
        else:
            print("Using full window (no region selected).")
    else:
        print("Using full window.")
    
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
