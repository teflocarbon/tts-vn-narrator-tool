#!/usr/bin/env python3
"""
TTS Visual Novel Narrator Tool - macOS Only

This application captures a selected window on macOS, monitors for changes,
and uses Apple's macOS OCR OCR to extract text when the content stops scrolling.
"""

import cv2
import numpy as np
import time
import threading
import os
from typing import Optional
import argparse

# macOS-specific imports
from window_capture import WindowCapture

# TTS integration
from tts_engine import initialize_tts, speak_text, stop_speaking, is_speaking

# OCR processing
from ocr_processor import OCRProcessor

# Logging
from logger import setup_logger, log_detected_text, log_monitor_status, log_image_difference


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
        
        # State management
        self.waiting_for_change = False
        self.last_image_hash = None
        
        # OCR processor
        self.ocr_processor = OCRProcessor(debug=debug, similarity_threshold=0.8)
        
        # TTS integration
        self.tts_enabled = True
        self.tts_engine = None
        
        # Setup logger
        self.logger = setup_logger('monitor', 'DEBUG' if debug else 'INFO')
        
        if debug:
            log_monitor_status("Debug mode enabled - Images will be saved to debug_captures/", "info")
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
        
        # Log the image difference with rich formatting
        is_different = difference_ratio > threshold
        log_image_difference(difference_ratio, threshold, is_different)
        
        return is_different
    
    def get_image_hash(self, image: np.ndarray) -> str:
        """
        Generate a simple hash for an image to detect changes.
        
        Args:
            image: Image to hash
            
        Returns:
            Hash string
        """
        if image is None:
            return ""
        
        # Simple hash based on image mean and shape
        return f"{image.shape}_{np.mean(image):.6f}"
    
    def on_text_detected(self, text: str):
        """
        Handle detected text with TTS integration.
        Uses OCRProcessor to check if text should be processed.
        
        Args:
            text: The detected text
        """
        # Use OCR processor to determine if text should be processed
        should_process, processed_text = self.ocr_processor.process_text_if_different(text)
        
        if should_process:
            # Display the detected text with rich formatting
            log_detected_text(processed_text)
            
            # TTS integration - speak the detected text
            if self.tts_enabled:
                try:
                    # Initialize TTS engine if not already done
                    if self.tts_engine is None:
                        log_monitor_status("Initializing TTS engine...", "info")
                        self.tts_engine = initialize_tts()
                    
                    # Stop any currently playing audio before speaking new text
                    stop_speaking()
                    
                    # Speak the new text
                    speak_text(processed_text)
                    
                except Exception as e:
                    log_monitor_status(f"TTS Error: {e}", "error")
                    log_monitor_status("Continuing without TTS...", "warning")
    
    def monitor_window(self):
        """
        Monitor the selected window for changes and extract text when stable.
        Optimized to only run OCR when the image actually changes.
        """
        log_monitor_status("Starting window monitoring...", "info")
        log_monitor_status("Press Ctrl+C to stop monitoring", "info")
        
        while self.monitoring:
            try:
                # Capture current window
                current_image = self.capture_current_window()
                
                if current_image is None:
                    log_monitor_status("Failed to capture window", "error")
                    time.sleep(self.check_interval)
                    continue
                
                # Generate hash for current image
                current_hash = self.get_image_hash(current_image)
                
                # Debug: Save frames periodically
                if self.debug and hasattr(self, 'debug_counter'):
                    if self.debug_counter % 10 == 0:  # Save every 10th frame
                        debug_path = f"debug_captures/frame_{self.debug_counter:04d}.png"
                        cv2.imwrite(debug_path, current_image)
                        self.logger.debug(f"Saved frame to {debug_path}")
                    self.debug_counter += 1
                elif self.debug:
                    self.debug_counter = 0
                
                # Compare with previous image
                if self.previous_image is not None:
                    # Check if image has changed
                    if self.images_are_different(current_image, self.previous_image):
                        # Image changed - reset waiting state and wait for stability
                        if self.waiting_for_change:
                            log_monitor_status("Change detected after stability - checking for new text", "info")
                            # We were waiting for change and got it - run OCR to see if text changed
                            text = self.ocr_processor.extract_text_with_macos_ocr(current_image)
                            if text:
                                self.on_text_detected(text)
                        else:
                            log_monitor_status("Change detected - waiting for stability...", "info")
                        
                        # Reset waiting state since we're seeing changes
                        self.waiting_for_change = False
                    else:
                        # No change detected - image is stable
                        if not self.waiting_for_change:
                            # Content has stabilized and we haven't processed it yet - run OCR
                            log_monitor_status("Content stabilized - running OCR", "info")
                            text = self.ocr_processor.extract_text_with_macos_ocr(current_image)
                            if text:
                                self.on_text_detected(text)
                            # Set flag to wait for next change
                            self.waiting_for_change = True
                        # If waiting_for_change is True, we've already processed this stable image
                        # so we don't need to do anything until the image changes
                else:
                    # First image capture
                    log_monitor_status("Initial capture - waiting for next frame...", "info")
                
                # Update previous image and hash
                self.previous_image = current_image.copy()
                self.last_image_hash = current_hash
                
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
            log_monitor_status("Already monitoring!", "warning")
            return
            
        # Check if we have a window selected
        if not self.window_capture or not self.window_capture.target_window:
            log_monitor_status("No window selected!", "error")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_window, daemon=True)
        self.monitor_thread.start()
        log_monitor_status("Monitoring started in background thread", "success")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        if not self.monitoring:
            return
            
        log_monitor_status("Stopping monitoring...", "info")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        log_monitor_status("Monitoring stopped", "success")


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
        log_monitor_status("No window selected. Exiting.", "error")
        return
    
    log_monitor_status("Window selected successfully.", "success")
    
    # Optionally select a region within the window
    log_monitor_status("\nDo you want to select a specific region within the window?", "info")
    choice = input("Press 'y' for region selection, or any other key to use the full window: ").strip().lower()
    
    if choice == 'y':
        window_region = monitor.select_window_region()
        if window_region:
            monitor.window_capture.set_region(window_region)
            log_monitor_status(f"Selected window region: {window_region}", "success")
        else:
            log_monitor_status("Using full window (no region selected).", "info")
    else:
        log_monitor_status("Using full window.", "info")
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Keep the main thread alive
        while monitor.monitoring:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        log_monitor_status("\nStopping monitoring...", "warning")
        monitor.stop_monitoring()
        log_monitor_status("Goodbye!", "info")


if __name__ == "__main__":
    main()
