#!/usr/bin/env python3
"""
macOS Window Capture Module

This module provides functionality to capture specific windows on macOS using CoreGraphics.
It offers more stability than screen region selection by targeting specific application windows.
"""

import Quartz as QZ
import Quartz.CoreGraphics as CG
import AppKit as AK

import numpy as np
import numpy.typing as npt

import typing as t

Matcher = t.Callable[[dict[str, t.Any]], bool]


def _find_window(matcher: Matcher) -> list[dict]:
    """Find the window id of the window with the given name."""

    window_list = QZ.CGWindowListCopyWindowInfo(
        QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID
    )

    result = []

    for window in window_list:
        if matcher(window):
            result.append(window)

    return result


def new_name_matcher(name: str) -> Matcher:
    """Create a matcher function that matches windows by exact name."""
    def matcher(window: dict[str, t.Any]) -> bool:
        return window.get("kCGWindowName") == name

    return matcher


def new_partial_name_matcher(partial_name: str) -> Matcher:
    """Create a matcher function that matches windows by partial name."""
    def matcher(window: dict[str, t.Any]) -> bool:
        window_name = window.get("kCGWindowName", "")
        return partial_name.lower() in window_name.lower()

    return matcher


def new_app_name_matcher(app_name: str) -> Matcher:
    """Create a matcher function that matches windows by application name."""
    def matcher(window: dict[str, t.Any]) -> bool:
        owner_name = window.get("kCGWindowOwnerName", "")
        return app_name.lower() in owner_name.lower()

    return matcher


def list_windows() -> list[dict]:
    """List all available windows with their properties."""
    window_list = QZ.CGWindowListCopyWindowInfo(
        QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID
    )
    
    visible_windows = []
    for window in window_list:
        # Only include windows that have a name and are on screen
        if (window.get("kCGWindowName") and 
            window.get("kCGWindowIsOnscreen", False) and
            window.get("kCGWindowLayer", 0) == 0):  # Normal window layer
            visible_windows.append({
                "name": window.get("kCGWindowName", ""),
                "app": window.get("kCGWindowOwnerName", ""),
                "id": window.get("kCGWindowNumber", 0),
                "bounds": window.get("kCGWindowBounds", {}),
            })
    
    return visible_windows


def capture_window_by_name(name: str) -> npt.NDArray[np.uint8]:
    """Capture a window by its exact name."""
    windows = _find_window(new_name_matcher(name))

    if len(windows) == 0:
        raise ValueError(f"Could not find window with name: {name}")
    elif len(windows) > 1:
        raise ValueError(f"Found multiple windows with name: {name}")

    window = windows[0]

    return _cg_capture_region_as_image(
        window_id=window["kCGWindowNumber"],
    )


def capture_window_by_partial_name(partial_name: str) -> npt.NDArray[np.uint8]:
    """Capture a window by partial name match."""
    windows = _find_window(new_partial_name_matcher(partial_name))

    if len(windows) == 0:
        raise ValueError(f"Could not find window containing: {partial_name}")
    elif len(windows) > 1:
        # Show available options
        window_names = [w.get("kCGWindowName", "Unknown") for w in windows]
        raise ValueError(f"Found multiple windows containing '{partial_name}': {window_names}")

    window = windows[0]

    return _cg_capture_region_as_image(
        window_id=window["kCGWindowNumber"],
    )


def capture_window_by_app(app_name: str) -> npt.NDArray[np.uint8]:
    """Capture the main window of an application."""
    windows = _find_window(new_app_name_matcher(app_name))

    if len(windows) == 0:
        raise ValueError(f"Could not find window for app: {app_name}")
    elif len(windows) > 1:
        # Try to find the main window (usually the first one or one with a title)
        main_window = None
        for window in windows:
            if window.get("kCGWindowName"):  # Has a title
                main_window = window
                break
        
        if main_window is None:
            main_window = windows[0]  # Fallback to first window
        
        window = main_window
    else:
        window = windows[0]

    return _cg_capture_region_as_image(
        window_id=window["kCGWindowNumber"],
    )


def capture_window_region(window_name: str, region: tuple[int, int, int, int]) -> npt.NDArray[np.uint8]:
    """
    Capture a specific region of a window.
    
    Args:
        window_name: Name of the window to capture
        region: Tuple of (x, y, width, height) relative to the window
        
    Returns:
        Numpy array of the captured region
    """
    # First capture the entire window
    full_window = capture_window_by_name(window_name)
    
    # Extract the specified region
    x, y, width, height = region
    
    # Ensure region is within bounds
    window_height, window_width = full_window.shape[:2]
    x = max(0, min(x, window_width - 1))
    y = max(0, min(y, window_height - 1))
    width = min(width, window_width - x)
    height = min(height, window_height - y)
    
    # Extract the region
    region_image = full_window[y:y+height, x:x+width]
    
    return region_image


def _cg_capture_region_as_image(
    region: tuple[int, int, int, int] | None = None,
    window_id: int | None = None,
) -> npt.NDArray[np.uint8]:
    """Capture a region of the screen using CoreGraphics."""

    if window_id is not None and region is not None:
        raise ValueError("Only one of region or window_id must be specified")

    image: CG.CGImage | None = None

    if region is not None:
        cg_region = None

        if region is None:
            cg_region = CG.CGRectInfinite
        else:
            cg_region = CG.CGRectMake(*region)

        image = CG.CGWindowListCreateImage(
            cg_region,
            CG.kCGWindowListOptionOnScreenOnly,
            CG.kCGNullWindowID,
            CG.kCGWindowImageDefault,
        )
    elif window_id is not None:
        image = CG.CGWindowListCreateImage(
            CG.CGRectNull,
            CG.kCGWindowListOptionIncludingWindow,
            window_id,
            CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution,
        )
    else:
        raise ValueError("Either region or window_id must be specified")

    if image is None:
        raise ValueError("Could not capture image")

    bpr = CG.CGImageGetBytesPerRow(image)
    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)

    cg_dataprovider = CG.CGImageGetDataProvider(image)
    cg_data = CG.CGDataProviderCopyData(cg_dataprovider)

    np_raw_data = np.frombuffer(cg_data, dtype=np.uint8)

    return np.lib.stride_tricks.as_strided(
        np_raw_data,
        shape=(height, width, 3),
        strides=(bpr, 4, 1),
        writeable=True,
    )


class WindowCapture:
    """
    A class to handle window capture with region selection support.
    Integrates with the existing ScreenRegionMonitor for seamless operation.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.target_window = None
        self.window_region = None
        
    def select_window(self) -> bool:
        """
        Allow user to select a window to monitor.
        
        Returns:
            True if a window was selected, False otherwise
        """
        print("\nAvailable windows:")
        print("-" * 50)
        
        windows = list_windows()
        
        if not windows:
            print("No windows found!")
            return False
        
        # Display windows with numbers
        for i, window in enumerate(windows, 1):
            bounds = window["bounds"]
            print(f"{i:2d}. {window['app']} - {window['name']}")
            print(f"    Size: {bounds.get('Width', 0)}x{bounds.get('Height', 0)}")
        
        print("-" * 50)
        
        while True:
            try:
                choice = input("Select window number (or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    return False
                
                window_idx = int(choice) - 1
                
                if 0 <= window_idx < len(windows):
                    self.target_window = windows[window_idx]
                    print(f"Selected: {self.target_window['app']} - {self.target_window['name']}")
                    return True
                else:
                    print(f"Please enter a number between 1 and {len(windows)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                return False
    
    def capture_current_window(self) -> npt.NDArray[np.uint8]:
        """
        Capture the currently selected window.
        
        Returns:
            Numpy array of the captured window
        """
        if not self.target_window:
            raise ValueError("No window selected. Call select_window() first.")
        
        try:
            # Capture by window name
            image = capture_window_by_name(self.target_window["name"])
            
            # If a region is specified, extract it
            if self.window_region:
                x, y, width, height = self.window_region
                window_height, window_width = image.shape[:2]
                
                # Ensure region is within bounds
                x = max(0, min(x, window_width - 1))
                y = max(0, min(y, window_height - 1))
                width = min(width, window_width - x)
                height = min(height, window_height - y)
                
                image = image[y:y+height, x:x+width]
            
            return image
            
        except ValueError as e:
            # Window might have been closed or renamed, try to find it again
            print(f"Warning: {e}")
            print("Attempting to find window again...")
            
            # Try partial name match
            try:
                image = capture_window_by_partial_name(self.target_window["name"])
                if self.window_region:
                    x, y, width, height = self.window_region
                    window_height, window_width = image.shape[:2]
                    x = max(0, min(x, window_width - 1))
                    y = max(0, min(y, window_height - 1))
                    width = min(width, window_width - x)
                    height = min(height, window_height - y)
                    image = image[y:y+height, x:x+width]
                return image
            except ValueError:
                raise ValueError(f"Could not capture window: {self.target_window['name']}")
    
    def set_region(self, region: tuple[int, int, int, int]):
        """Set a specific region within the window to capture."""
        self.window_region = region
        print(f"Set capture region: {region}")
