Title: AI-Powered Video Translator with TTS and Subtitles

Description:
This Python project provides a command-line tool for translating videos, incorporating several powerful AI technologies. The tool takes a video as input, extracts the audio, transcribes it using the Whisper model, translates the text using the Google Translate API, refines the translation with Gemini API, generates Text-to-Speech (TTS) audio using various engines (gTTS, OpenAI TTS, and Google Cloud TTS), and finally combines the translated audio with the original video while generating an SRT subtitle file.

Key Features:
    Audio Extraction: Extracts audio from video files in chunks using FFmpeg for efficient processing.
    Transcription: Transcribes audio to text using the faster-whisper library.
    Translation: Translates transcribed text to the target language using deep-translator library (Google Translate).
    Text Refinement: Uses the Gemini API to refine the translated text for better subtitle quality.
    Text-to-Speech (TTS): Generates audio from translated text using gTTS, OpenAI TTS API, or Google Cloud TTS API.
    Video Combination: Combines the original video with the generated TTS audio using FFmpeg.
    SRT Subtitle Generation: Creates an SRT file with translated subtitles, synchronized with audio.
    Multiple TTS Engine Support: Support for different TTS engines: gTTS, OpenAI, Google Cloud TTS
    Command-Line Interface: Uses argparse for easy command-line execution.
    Logging: Detailed logging to track the progress and diagnose errors
    Error Handling: Robust error handling to manage exceptions during the video translation.
    Configuration: Easy setup with configurable parameters.

Technologies Used:
    Python 3.6+
    faster-whisper
    deep-translator
    gTTS
    google-generativeai
    google-cloud-texttospeech
    openai
    FFmpeg
    torch

Install
pip install faster-whisper deep-translator gtts tqdm google-generativeai google-cloud-texttospeech openai

Usage:

# Command Line Interface (CLI)
python ai-translate.py --video input.mp4 --output output.mp4 --srt subtitles.srt --language de --tts_engine gtts
