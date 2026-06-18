# speech_interface_v2.py
"""
SignalBot Speech Interface - PUSH-TO-TALK VERSION
--------------------------------------------------
Eliminates feedback loop by only listening when key is pressed.

TTS: pyttsx3 (offline, fast)
STT: SpeechRecognition with Google API (free tier)
Input: Push-to-talk (default: SPACE key)

NEW: No microphone feedback - mic only active when you hold the key!
"""

import sys
import time
import threading
from typing import Optional

# TTS engine
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[SPEECH] pyttsx3 not available. Install: pip install pyttsx3 --break-system-packages")

# STT engine
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("[SPEECH] SpeechRecognition not available. Install: pip install SpeechRecognition --break-system-packages")

# PyAudio check (needed for microphone)
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[SPEECH] PyAudio not available. Install: pip install pyaudio --break-system-packages")

# Keyboard for push-to-talk
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[SPEECH] keyboard not available. Install: pip install keyboard --break-system-packages")
    print("[SPEECH] (Required for push-to-talk functionality)")


class SpeechInterface:
    """
    Push-to-talk speech interface - eliminates feedback loop.
    
    Press and hold SPACE (or configured key) to speak.
    Release to stop recording and process speech.
    """
    
    def __init__(self, push_to_talk_key: str = 'space'):
        self.enabled = False
        self.tts_engine = None
        self.stt_recognizer = None
        self.microphone = None
        self.push_to_talk_key = push_to_talk_key
        
        self._init_tts()
        self._init_stt()
    
    def _init_tts(self):
        """Initialize text-to-speech engine."""
        if not TTS_AVAILABLE:
            return
        
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure for speed and clarity
            self.tts_engine.setProperty('rate', 180)  # Slightly faster than default
            self.tts_engine.setProperty('volume', 0.9)
            
            # Try to set a better voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'english' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            print(f"[SPEECH] TTS engine initialized | rate=180 wpm")
            
        except Exception as e:
            print(f"[SPEECH] TTS init failed: {e}")
            self.tts_engine = None
    
    def _init_stt(self):
        """Initialize speech-to-text."""
        if not STT_AVAILABLE or not PYAUDIO_AVAILABLE:
            return
        
        try:
            self.stt_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrate for ambient noise
            print("[SPEECH] Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.stt_recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("[SPEECH] STT engine initialized | using Google API")
            
        except Exception as e:
            print(f"[SPEECH] STT init failed: {e}")
            self.stt_recognizer = None
            self.microphone = None
    
    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════
    
    def enable(self) -> bool:
        """Enable speech mode. Returns True if successful."""
        if not self.is_available():
            print("[SPEECH] Cannot enable - engines not available")
            if not KEYBOARD_AVAILABLE:
                print("[SPEECH] Missing keyboard module - run: pip install keyboard --break-system-packages")
            return False
        
        self.enabled = True
        print(f"[SPEECH] Speech mode ENABLED (Push-to-talk: {self.push_to_talk_key.upper()})")
        return True
    
    def disable(self):
        """Disable speech mode."""
        self.enabled = False
        print("[SPEECH] Speech mode DISABLED")
    
    def is_enabled(self) -> bool:
        """Check if speech mode is active."""
        return self.enabled
    
    def is_available(self) -> bool:
        """Check if speech engines are available."""
        return (self.tts_engine is not None and 
                self.stt_recognizer is not None and 
                self.microphone is not None and
                KEYBOARD_AVAILABLE)
    
    def speak(self, text: str, blocking: bool = True):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to finish
        """
        if not self.enabled or not self.tts_engine:
            return
        
        try:
            # Clean text for speech
            clean_text = text.replace('*', '')  # Remove action markers
            clean_text = clean_text.replace('[GROUND]', '').replace('[DREAM]', '')
            clean_text = ''.join(c for c in clean_text if ord(c) < 0x1F600 or ord(c) > 0x1F64F)
            clean_text = clean_text.strip()
            
            if not clean_text:
                return
            
            if blocking:
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            else:
                # Non-blocking speech in separate thread
                def _speak_async():
                    self.tts_engine.say(clean_text)
                    self.tts_engine.runAndWait()
                
                thread = threading.Thread(target=_speak_async, daemon=True)
                thread.start()
            
        except Exception as e:
            print(f"[SPEECH] TTS error: {e}")
    
    def listen_push_to_talk(self, max_wait: float = 30.0) -> Optional[str]:
        """
        Push-to-talk listening mode.
        
        User presses and holds the configured key to speak.
        Release key to stop recording and process speech.
        
        Args:
            max_wait: Maximum seconds to wait for key press
        
        Returns:
            Recognized text or None if cancelled/timeout
        """
        if not self.enabled or not self.stt_recognizer or not self.microphone or not KEYBOARD_AVAILABLE:
            return None
        
        try:
            # Wait for key press
            print(f"[SPEECH] Press and HOLD {self.push_to_talk_key.upper()} to speak (or type to skip)...")
            
            start_time = time.time()
            
            # Wait for key press with timeout
            while True:
                if keyboard.is_pressed(self.push_to_talk_key):
                    break
                
                # Check timeout
                if time.time() - start_time > max_wait:
                    print("[SPEECH] Timeout - no key pressed")
                    return None
                
                time.sleep(0.05)  # Small delay to prevent CPU spinning
            
            print(f"[SPEECH] Recording... (release {self.push_to_talk_key.upper()} when done)")
            
            # Record while key is held
            with self.microphone as source:
                # Start recording
                frames = []
                
                # Record in chunks while key is pressed
                while keyboard.is_pressed(self.push_to_talk_key):
                    try:
                        # Record small chunk
                        audio_chunk = self.stt_recognizer.listen(source, timeout=0.5, phrase_time_limit=0.5)
                        frames.append(audio_chunk)
                    except sr.WaitTimeoutError:
                        # No audio in this chunk, continue
                        continue
                
                print("[SPEECH] Processing audio...")
                
                # If we got no audio, return None
                if not frames:
                    print("[SPEECH] No audio recorded")
                    return None
                
                # Combine audio frames (use the longest one for simplicity)
                # In a more sophisticated version, we'd concatenate properly
                audio = max(frames, key=lambda x: len(x.frame_data)) if frames else None
                
                if not audio:
                    return None
            
            # Try to recognize using Google Speech API
            try:
                text = self.stt_recognizer.recognize_google(audio)
                print(f"[SPEECH] Recognized: {text}")
                return text
            
            except sr.UnknownValueError:
                print("[SPEECH] Could not understand audio")
                return None
            
            except sr.RequestError as e:
                print(f"[SPEECH] Google API error: {e}")
                return None
        
        except Exception as e:
            print(f"[SPEECH] Listen error: {e}")
            return None
    
    def get_status(self) -> str:
        """Get status report for debugging."""
        status = "[SPEECH STATUS]\n"
        status += f"  Mode: Push-to-Talk\n"
        status += f"  Key: {self.push_to_talk_key.upper()}\n"
        status += f"  Enabled: {self.enabled}\n"
        status += f"  TTS Available: {self.tts_engine is not None}\n"
        status += f"  STT Available: {self.stt_recognizer is not None}\n"
        status += f"  Microphone Available: {self.microphone is not None}\n"
        status += f"  Keyboard Available: {KEYBOARD_AVAILABLE}\n"
        
        if self.tts_engine:
            try:
                rate = self.tts_engine.getProperty('rate')
                volume = self.tts_engine.getProperty('volume')
                status += f"  TTS Rate: {rate} wpm\n"
                status += f"  TTS Volume: {volume:.1f}\n"
            except:
                pass
        
        return status


# ═══════════════════════════════════════════════════════════
# GLOBAL SINGLETON
# ═══════════════════════════════════════════════════════════

_speech_interface: Optional[SpeechInterface] = None

def get_speech_interface(push_to_talk_key: str = 'space') -> SpeechInterface:
    """Get or create the global speech interface."""
    global _speech_interface
    if _speech_interface is None:
        _speech_interface = SpeechInterface(push_to_talk_key=push_to_talk_key)
    return _speech_interface

# Convenience functions
def speak(text: str, blocking: bool = True):
    """Speak text if speech is enabled."""
    get_speech_interface().speak(text, blocking)

def listen_push_to_talk(max_wait: float = 30.0) -> Optional[str]:
    """Listen for voice input using push-to-talk."""
    return get_speech_interface().listen_push_to_talk(max_wait)

def enable_speech(push_to_talk_key: str = 'space') -> bool:
    """Enable speech mode."""
    return get_speech_interface(push_to_talk_key).enable()

def disable_speech():
    """Disable speech mode."""
    get_speech_interface().disable()

def is_speech_enabled() -> bool:
    """Check if speech is enabled."""
    return get_speech_interface().is_enabled()

def get_status() -> str:
    """Get speech status."""
    return get_speech_interface().get_status()


# ═══════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════

def test_push_to_talk():
    """Test push-to-talk functionality."""
    print("\n=== Testing Push-to-Talk ===")
    speech = get_speech_interface()
    
    if not speech.is_available():
        print("Speech system not fully available")
        return False
    
    speech.enable()
    
    print("\nTest 1: Simple recording")
    print("When prompted, hold SPACE and say: 'Hello SignalBot'")
    result = speech.listen_push_to_talk(max_wait=10.0)
    
    if result:
        print(f"✓ Recognition successful: {result}")
        speech.speak(f"You said: {result}")
        return True
    else:
        print("✗ No speech recognized")
        return False

def run_tests():
    """Run all tests."""
    print("SignalBot Push-to-Talk Speech Interface")
    print("=" * 50)
    
    speech = get_speech_interface()
    print(speech.get_status())
    
    if not speech.is_available():
        print("\nSpeech system not fully available. Check installation:")
        if not TTS_AVAILABLE:
            print("  - Missing pyttsx3")
        if not STT_AVAILABLE:
            print("  - Missing SpeechRecognition")
        if not PYAUDIO_AVAILABLE:
            print("  - Missing pyaudio")
        if not KEYBOARD_AVAILABLE:
            print("  - Missing keyboard (pip install keyboard --break-system-packages)")
        return
    
    # Test push-to-talk
    success = test_push_to_talk()
    
    print("\n" + "=" * 50)
    print(f"Push-to-Talk Test: {'PASS' if success else 'FAIL'}")


if __name__ == "__main__":
    run_tests()
