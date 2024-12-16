<h1>Title: AI-Powered Video Translator with TTS and Subtitles SRT</h1>

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
Generate a Gemini Key and add it. Thats all. If you want to use other TTS engine you need to also API Keys.
Usage:

# Command Line Interface (CLI)

Basic Translation with Default Settings
This command uses the default settings defined in your script for the medium model, gtts as the TTS engine, and translates the video to German (de)
      python ai-translate.py --video input.mp4 --output translated_output.mp4 --srt subtitles.srt --language de
sample German: python ai-translate.py --video input.mp4 --output output.mp4 --srt subtitles.srt --language de --tts_engine gtts
    --video input.mp4: Specifies the input video file. (Replace input.mp4 with your video file's name).
•	--output translated_output.mp4: Specifies the output video file with the translated audio.
•	--srt subtitles.srt: Specifies the output SRT subtitle file.
•	--language de: Specifies the target language for translation (German).

Using a Different Target Language
To translate to Spanish (es):
      python ai-translate.py --video input.mp4 --output translated_spanish.mp4 --srt subtitles_es.srt --language es
    
To translate to French (fr):
      python ai-translate.py --video input.mp4 --output translated_french.mp4 --srt subtitles_fr.srt --language fr
    
To translate to Italian (it):
      python ai-translate.py --video input.mp4 --output translated_italian.mp4 --srt subtitles_it.srt --language it
    
To translate to Japanese (ja):
      python ai-translate.py --video input.mp4 --output translated_japanese.mp4 --srt subtitles_ja.srt --language ja
    
To translate to Korean (ko):
      python ai-translate.py --video input.mp4 --output translated_korean.mp4 --srt subtitles_ko.srt --language ko
    
Using the OpenAI TTS Engine
To use the OpenAI TTS engine, you will need to set the OPENAI_API_KEY in your environment variables.
      python ai-translate.py --video input.mp4 --output openai_translated.mp4 --srt openai_subtitles.srt --language de --tts_engine openai
    
Custom OpenAI Voice and Speed
To specify a different voice and speed with OpenAI:
      python ai-translate.py --video input.mp4 --output openai_custom.mp4 --srt openai_custom.srt --language de --tts_engine openai --openai_voice nova --openai_speed 1.2
    
•	--openai_voice nova: Sets the OpenAI voice to nova.
•	--openai_speed 1.2: Sets the OpenAI speed to 1.2 (you can use values from 0.25 to 4.0).

