"""
Centralized logging configuration using Rich library for the TTS Visual Novel Narrator Tool.
"""

import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.panel import Panel
from typing import Optional

# Global console instance
console = Console()

# Configure logging levels
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Set up a logger with Rich formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Create Rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True
    )
    
    # Set format
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    rich_handler.setFormatter(formatter)
    
    logger.addHandler(rich_handler)
    logger.propagate = False
    
    return logger


def log_detected_text(text: str):
    """
    Log detected text with special formatting.
    
    Args:
        text: The detected text to display
    """
    panel = Panel(
        Text(text, style="bold white"),
        title="[bold green]DETECTED TEXT[/bold green]",
        border_style="green",
        padding=(1, 2)
    )
    console.print(panel)


def log_tts_status(message: str, status: str = "info"):
    """
    Log TTS-related status messages.
    
    Args:
        message: Status message
        status: Status type ('info', 'success', 'warning', 'error')
    """
    styles = {
        'info': 'blue',
        'success': 'green',
        'warning': 'yellow',
        'error': 'red'
    }
    
    style = styles.get(status, 'blue')
    console.print(f"[{style}]🔊 TTS:[/{style}] {message}")


def log_ocr_status(message: str, status: str = "info"):
    """
    Log OCR-related status messages.
    
    Args:
        message: Status message
        status: Status type ('info', 'success', 'warning', 'error')
    """
    styles = {
        'info': 'cyan',
        'success': 'green',
        'warning': 'yellow',
        'error': 'red'
    }
    
    style = styles.get(status, 'cyan')
    console.print(f"[{style}]👁️  OCR:[/{style}] {message}")


def log_monitor_status(message: str, status: str = "info"):
    """
    Log monitoring-related status messages.
    
    Args:
        message: Status message
        status: Status type ('info', 'success', 'warning', 'error')
    """
    styles = {
        'info': 'magenta',
        'success': 'green',
        'warning': 'yellow',
        'error': 'red'
    }
    
    style = styles.get(status, 'magenta')
    console.print(f"[{style}]📱 Monitor:[/{style}] {message}")


def log_similarity_check(similarity: float, threshold: float, skipped: bool = False):
    """
    Log text similarity check results.
    
    Args:
        similarity: Calculated similarity ratio
        threshold: Similarity threshold
        skipped: Whether the text was skipped due to high similarity
    """
    if skipped:
        console.print(f"[yellow]⚠️  Similarity:[/yellow] {similarity:.2f} >= {threshold:.2f} - [red]skipping output[/red]")
    else:
        console.print(f"[green]✅ Similarity:[/green] {similarity:.2f} < {threshold:.2f} - [green]processing text[/green]")


def log_image_difference(ratio: float, threshold: float, changed: bool = False):
    """
    Log image difference analysis results.
    
    Args:
        ratio: Image difference ratio
        threshold: Difference threshold
        changed: Whether the image was considered changed
    """
    if changed:
        console.print(f"[green]📸 Image:[/green] difference {ratio:.4f} >= {threshold:.4f} - [green]change detected[/green]")
    else:
        console.print(f"[dim]📸 Image:[/dim] difference {ratio:.4f} < {threshold:.4f} - [dim]no change[/dim]")


# Create module-level loggers
main_logger = setup_logger('main')
tts_logger = setup_logger('tts')
ocr_logger = setup_logger('ocr')
monitor_logger = setup_logger('monitor')
