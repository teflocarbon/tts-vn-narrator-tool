"""
TTS Engine for Visual Novel Narrator Tool

This module provides text-to-speech functionality using OpenAI's TTS API
with a local server (SparkTTS or similar).
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from openai import OpenAI
import pygame
from logger import setup_logger, log_tts_status


class TTSEngine:
    def __init__(self, base_url="http://localhost:9991/v1", voice="default", model="tts-1"):
        """
        Initialize the TTS engine.
        
        Args:
            base_url (str): Base URL for the TTS API server
            voice (str): Voice to use for TTS
            model (str): Model to use for TTS
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"
        )
        self.voice = voice
        self.model = model
        self.is_speaking = False
        self.current_thread = None
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Create temp directory for audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "tts_narrator"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger('tts')
    
    def speak(self, text):
        """
        Convert text to speech and play it.
        
        Args:
            text (str): Text to convert to speech
        """
        if not text.strip():
            return
        
        # Stop any currently playing audio
        self.stop()
        
        # Start speaking in a separate thread to avoid blocking
        self.current_thread = threading.Thread(target=self._speak_async, args=(text,))
        self.current_thread.daemon = True
        self.current_thread.start()
    
    def _speak_async(self, text):
        """
        Asynchronously convert text to speech and play it.
        
        Args:
            text (str): Text to convert to speech
        """
        try:
            self.is_speaking = True
            
            # Generate unique filename for this audio
            timestamp = int(time.time() * 1000)
            audio_file = self.temp_dir / f"speech_{timestamp}.wav"
            
            log_tts_status(f"Generating TTS for: {text[:50]}...", "info")
            
            # Generate speech using OpenAI API
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text
            ) as response:
                response.stream_to_file(str(audio_file))
            
            # Play the audio file
            if audio_file.exists():
                log_tts_status(f"Playing audio: {audio_file.name}", "success")
                pygame.mixer.music.load(str(audio_file))
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    if not self.is_speaking:  # Allow interruption
                        break
                
                # Clean up the temporary file
                try:
                    audio_file.unlink()
                except OSError:
                    pass  # File might be in use, ignore
            
        except Exception as e:
            log_tts_status(f"TTS Error: {e}", "error")
        finally:
            self.is_speaking = False
    
    def stop(self):
        """
        Stop any currently playing audio.
        """
        self.is_speaking = False
        pygame.mixer.music.stop()
        
        # Wait for current thread to finish
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=1.0)
    
    def is_busy(self):
        """
        Check if TTS is currently speaking.
        
        Returns:
            bool: True if currently speaking, False otherwise
        """
        return self.is_speaking or pygame.mixer.music.get_busy()
    
    def set_voice(self, voice):
        """
        Change the TTS voice.
        
        Args:
            voice (str): New voice to use
        """
        self.voice = voice
    
    def cleanup(self):
        """
        Clean up resources and temporary files.
        """
        self.stop()
        
        # Clean up any remaining temp files
        try:
            for file in self.temp_dir.glob("speech_*.wav"):
                file.unlink()
        except OSError:
            pass


# Global TTS engine instance
tts_engine = None


def initialize_tts(base_url="http://10.0.10.240:9991/v1", voice="default", model="tts-1"):
    """
    Initialize the global TTS engine.
    
    Args:
        base_url (str): Base URL for the TTS API server
        voice (str): Voice to use for TTS
        model (str): Model to use for TTS
    
    Returns:
        TTSEngine: The initialized TTS engine
    """
    global tts_engine
    tts_engine = TTSEngine(base_url=base_url, voice=voice, model=model)
    return tts_engine


def speak_text(text):
    """
    Convenience function to speak text using the global TTS engine.
    
    Args:
        text (str): Text to speak
    """
    global tts_engine
    if tts_engine is None:
        tts_engine = initialize_tts()
    
    tts_engine.speak(text)


def stop_speaking():
    """
    Stop any currently playing TTS audio.
    """
    global tts_engine
    if tts_engine:
        tts_engine.stop()


def is_speaking():
    """
    Check if TTS is currently active.
    
    Returns:
        bool: True if speaking, False otherwise
    """
    global tts_engine
    if tts_engine:
        return tts_engine.is_busy()
    return False


if __name__ == "__main__":
    # Test the TTS engine
    print("Testing TTS Engine...")
    
    # Initialize TTS
    engine = initialize_tts()
    
    # Test speech
    test_text = "Hello from the TTS Visual Novel Narrator Tool! This is a test of the text-to-speech functionality."
    print(f"Speaking: {test_text}")
    speak_text(test_text)
    
    # Wait for completion
    while is_speaking():
        time.sleep(0.1)
    
    print("TTS test completed!")
    engine.cleanup()
