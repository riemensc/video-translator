import os
import subprocess
import torch
import tempfile
import logging
from faster_whisper import WhisperModel
import argparse
from tqdm import tqdm
from deep_translator import GoogleTranslator
from gtts import gTTS
import shutil
import google.generativeai as genai
from google.cloud import texttospeech
import openai

# --- Configuration ---
VIDEO_FILE = "start.mp4"  # Replace with your video file
OUTPUT_VIDEO_FILE = "translated_video.mp4"
SRT_FILE = "subtitles.srt"  # Name of the output SRT file
MODEL_SIZE = "medium"
TARGET_LANGUAGE = "de"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_DIR = "temp_files"
FFMPEG_TIMEOUT = 600  # Timeout for ffmpeg processes in seconds
AUDIO_EXTRACTION_CHUNK_SIZE = 30  # Chunk size for audio extraction (adjust as needed)
TRANSCRIBE_CHUNK_SIZE = 300  # Chunk size for transcription
GEMINI_API_KEY = " Your Gemini API key"  # Your Gemini API key
# Google Cloud TTS
GOOGLE_CLOUD_TTS_KEY_FILE = "path/to/your/google_cloud_credentials.json"  # Path to your Google Cloud service account key file
GOOGLE_CLOUD_VOICE = "en-US-Neural2-F"  # Default Google Cloud voice
GOOGLE_CLOUD_SPEED = 1.0  # Default Google Cloud speech speed (0.25 to 4.0)
# OpenAI TTS
OPENAI_API_KEY = "YOUR-OPENAPI-KEY"
OPENAI_VOICE = "onyx"  # Default OpenAI voice
OPENAI_SPEED = 1.0  # Default OpenAI speech speed (0.25 to 4.0)
TTS_ENGINE = "gtts"  # Default TTS engine: "openai", "google", or "gtts"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Exceptions ---
class TranslationError(Exception):
    pass

class TTSError(Exception):
    pass

class AudioExtractionError(Exception):
    pass

class FFmpegError(Exception):
    pass

class SubtitleError(Exception):
    pass

class GeminiError(Exception):  # For Gemini API errors
    pass

# --- Utility Functions ---

def format_time(seconds):
    """Formats time into SRT-compatible format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

def safe_filename(filename):
    """Sanitizes a filename by removing or replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*\0'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# --- Core Functions ---

def extract_audio_chunk(video_path, audio_path, start_time, duration, timeout=FFMPEG_TIMEOUT):
    """Extracts a chunk of audio from the video to MP3."""
    logging.info(
        f"Extracting audio chunk from '{video_path}' to '{audio_path}' (start: {start_time}, duration: {duration})..."
    )
    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-threads", "auto",  # Enable multi-threading
                # "-hwaccel", "nvdec", # Uncomment for hardware acceleration (if supported)
                "-ss",
                str(start_time),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-map",
                "0:a",
                "-c:a",
                "libmp3lame",  # MP3 encoding
                "-q:a", "2",  # Good quality MP3 (VBR)
                "-loglevel",
                "error",
                audio_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = process.communicate(timeout=timeout)

        if process.returncode != 0:
            logging.error(f"ffmpeg exited with error code {process.returncode}:")
            logging.error(stderr.decode())
            raise FFmpegError(f"ffmpeg exited with error code {process.returncode}")

        logging.info(f"Audio chunk successfully extracted: '{audio_path}'.")

    except subprocess.TimeoutExpired:
        process.kill()
        logging.error(f"ffmpeg process timed out after {timeout} seconds.")
        raise FFmpegError(f"ffmpeg process timed out during audio extraction.")

    except FFmpegError as e:
        raise
    except Exception as e:
        logging.error(f"Error extracting audio chunk: {e}")
        raise AudioExtractionError(f"Error extracting audio chunk: {e}")

def get_video_duration(video_path):
    """Gets the duration of the video in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting video duration: {e}")
        raise

def extract_audio_in_chunks(video_path, audio_output_dir, chunk_size=AUDIO_EXTRACTION_CHUNK_SIZE):
    """Extracts audio from a video in chunks (MP3 format)."""
    logging.info(f"Extracting audio from '{video_path}' in chunks of {chunk_size} seconds...")
    try:
        total_duration = get_video_duration(video_path)
        logging.info(f"Total video duration: {total_duration} seconds")

        audio_chunks = []
        for start_time in range(0, int(total_duration), chunk_size):
            end_time = min(start_time + chunk_size, total_duration)
            duration = end_time - start_time
            chunk_filename = f"audio_chunk_{start_time}-{end_time}.mp3"  # MP3 extension
            chunk_path = os.path.join(audio_output_dir, chunk_filename)

            logging.info(f"Extracting chunk starting at {start_time}...")
            extract_audio_chunk(video_path, chunk_path, start_time, duration)
            logging.info(f"Finished extracting chunk starting at {start_time}.")

            audio_chunks.append(chunk_path)

        logging.info(f"Audio extraction completed. {len(audio_chunks)} chunks extracted.")
        return audio_chunks, total_duration

    except Exception as e:
        logging.error(f"Error extracting audio in chunks: {e}")
        raise AudioExtractionError(f"Error extracting audio in chunks: {e}")

def merge_audio_chunks(audio_chunks, output_path, timeout=FFMPEG_TIMEOUT):
    """Merges multiple audio chunks into a single file using ffmpeg."""
    logging.info(f"Merging {len(audio_chunks)} audio chunks into '{output_path}'...")

    if not audio_chunks:
        logging.error("No audio chunks provided to merge.")
        raise ValueError("No audio chunks provided for merging.")

    # Create a text file listing the audio chunk files
    list_file_path = os.path.join(os.path.dirname(output_path), "chunk_list.txt")
    with open(list_file_path, "w") as f:
        for chunk_path in audio_chunks:
            f.write(f"file '{os.path.abspath(chunk_path)}'\n")

    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file_path,
                "-c",
                "copy",
                "-loglevel",
                "error",
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = process.communicate(timeout=timeout)

        if process.returncode != 0:
            logging.error(f"ffmpeg exited with error code {process.returncode}:")
            logging.error(stderr.decode())
            raise FFmpegError(f"ffmpeg exited with error code {process.returncode} during audio merging")

        logging.info(f"Audio chunks successfully merged into '{output_path}'.")

    except subprocess.TimeoutExpired:
        process.kill()
        logging.error(f"ffmpeg process timed out after {timeout} seconds during merging.")
        raise FFmpegError(f"ffmpeg process timed out during audio merging.")

    except FFmpegError as e:
        raise
    except Exception as e:
        logging.error(f"Error merging audio chunks: {e}")
        raise AudioExtractionError(f"Error merging audio chunks: {e}")
    finally:
        os.remove(list_file_path)  # Clean up the list file

def transcribe_audio_in_chunks(audio_chunks, model, language="en", beam_size=5, chunk_size=TRANSCRIBE_CHUNK_SIZE):
    """Transcribes audio in chunks using Whisper."""
    logging.info(f"Starting transcription of audio chunks using Whisper (model: {MODEL_SIZE})...")
    all_segments = []
    total_words_original = 0
    try:
        for audio_chunk_path in tqdm(audio_chunks, desc="Transcribing audio chunks"):
            # Extract start time from audio chunk filename
            start_time_str = audio_chunk_path.split("_")[-1].split(".")[0].split("-")[0]
            if start_time_str.isdigit():
                start_time = int(start_time_str)
            else:
                logging.warning(
                    f"Could not extract valid start time from filename: {audio_chunk_path}. Using 0 as default."
                )
                start_time = 0

            logging.info(f"Transcribing chunk: {audio_chunk_path} (Starting at: {start_time}s)...")
            segments, info = model.transcribe(audio_chunk_path, beam_size=beam_size, language=language)
            for segment in segments:
                text = segment.text
                word_count = len(text.split())
                total_words_original += word_count
                all_segments.append({
                    "start": start_time + segment.start,
                    "end": start_time + segment.end,
                    "text": text,
                    "word_count": word_count
                })
                logging.info(f"Segment: '{text}' (Words: {word_count})")
            logging.info(f"Finished transcribing chunk: {audio_chunk_path}")

        logging.info(f"Transcription completed. Total segments: {len(all_segments)}")
        logging.info(f"Total words in original transcript: {total_words_original}")
        return all_segments, total_words_original

    except Exception as e:
        logging.error(f"Transcription error: {e}")
        raise

def translate_text(text, target_language):
    """Translates text using deep-translator."""
    try:
        translator = GoogleTranslator(source="auto", target=target_language)
        translated_text = translator.translate(text)
        logging.info(f"Original: '{text}' --> Translated: '{translated_text}'")
        return translated_text
    except Exception as e:
        logging.warning(f"Translation error for '{text}': {e}. Using original text.")
        return text

def create_subtitle_file(segments, subtitle_path, target_language):
    """Creates an SRT file with translated subtitles."""
    logging.info(f"Creating subtitle file '{subtitle_path}'...")
    try:
        with open(subtitle_path, "w", encoding="utf-8") as f:
            for i, segment in tqdm(enumerate(segments), total=len(segments), desc="Generating subtitles"):
                translated_text = segment["translated_text"]
                f.write(f"{i + 1}\n{format_time(segment['start'])} --> {format_time(segment['end'])}\n{translated_text}\n\n")
        logging.info(f"Subtitle file created: {subtitle_path}")
        return True
    except Exception as e:
        logging.error(f"Error creating subtitle file: {e}")
        raise SubtitleError(f"Error creating subtitle file: {e}")

def get_mp3_duration(file_path):
    """Gets the duration of an MP3 file using ffprobe."""
    try:
        duration_output = subprocess.check_output([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path
        ], text=True)
        duration = float(duration_output.strip())

        return duration

    except Exception as e:
        logging.error(f"Error getting duration for {file_path}: {e}")
        return 0  # Or raise the exception, depending on your error handling

def refine_text_with_gemini(translated_segments, target_language, total_words_original):
    """Refines all translated text segments using the Gemini API."""
    logging.info("Starting bulk text refinement with Gemini...")
    refined_segments = []
    total_words_translated = 0
    total_words_refined = 0
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")

        all_translated_text = " ".join([segment["translated_text"] for segment in translated_segments])
        total_words_translated = len(all_translated_text.split())

        prompt = (
            f"Refine this translated text to be as close as possible to {total_words_original} words "
            f"while preserving the meaning of the original text. "
            f"Make the refined text suitable for use as subtitles in a video. "
            f"Translated text to refine: '{all_translated_text}' "
            f"Target language: {target_language}"
        )

        response = model.generate_content(prompt)

        if response.text:
            refined_text = response.text
            total_words_refined = len(refined_text.split())
            logging.info(f"Refined text: '{refined_text}'")

            # Calculate and log the percentage change in word count
            percentage_change = ((total_words_refined - total_words_original) / total_words_original) * 100
            logging.info(f"Total words (Original): {total_words_original}")
            logging.info(f"Total words (Translated): {total_words_translated}")
            logging.info(f"Total words (Refined): {total_words_refined}")
            logging.info(f"Percentage change in word count: {percentage_change:.2f}%")

            refined_segments = split_refined_text(translated_segments, refined_text)

        else:
            logging.warning("Gemini did not return a refined text. Using the original translated segments.")
            refined_segments = translated_segments

        return refined_segments

    except Exception as e:
        logging.error(f"Error refining text with Gemini: {e}")
        raise GeminiError(f"Error refining text with Gemini: {e}")

def split_refined_text(original_segments, refined_text):
    """Splits the refined text back into segments based on the original segment boundaries."""
    refined_segments = []
    current_index = 0

    for segment in original_segments:
        original_length = len(segment["text"])

        # Find the end of a sentence closest to the original segment length
        segment_end_index = refined_text.find(".", current_index) + 1

        # If no period is found, take the rest of the text
        if segment_end_index == 0:
            segment_end_index = len(refined_text)

        new_text = refined_text[current_index:segment_end_index].strip()

        refined_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],  # Keep the original text for reference
            "word_count": len(new_text.split()),  # Update with refined word count
            "translated_text": new_text
        })

        # Move the index to the start of the next segment
        current_index = segment_end_index

    return refined_segments

def generate_tts_audio_with_openai(text, segment_index, temp_dir, language, voice, speed):
    """Generates TTS audio using OpenAI's TTS API."""
    try:
        tts_filename = f"tts_segment_{segment_index}.mp3"
        tts_filepath = os.path.join(temp_dir, tts_filename)
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Convert language code for OpenAI (assuming Whisper uses ISO 639-1 codes)
        if language == "de":
            openai_language = "de"
        elif language == "fr":
            openai_language = "fr"
        elif language == "es":
            openai_language = "es"
        elif language == "it":
            openai_language = "it"
        elif language == "ja":
            openai_language = "ja"
        elif language == "ko":
            openai_language = "ko"
        else:
            openai_language = "en"  # Default to English

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed
        )
        response.stream_to_file(tts_filepath)

        duration = get_mp3_duration(tts_filepath)

        logging.info(
            f"TTS audio generated with OpenAI for segment {segment_index + 1} (Voice: {voice}, Speed: {speed}, Language: {openai_language}).")
        return tts_filepath, duration

    except Exception as e:
        logging.error(f"Error generating TTS with OpenAI for segment {segment_index + 1}: {e}")
        raise TTSError(f"Error generating TTS with OpenAI: {e}")

def generate_tts_audio_with_google_cloud(text, segment_index, temp_dir, language, voice_name, speed):
    """Generates TTS audio using Google Cloud Text-to-Speech API."""
    try:
        tts_filename = f"tts_segment_{segment_index}.mp3"
        tts_filepath = os.path.join(temp_dir, tts_filename)

        client = texttospeech.TextToSpeechClient.from_service_account_file(GOOGLE_CLOUD_TTS_KEY_FILE)

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language, name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speed
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(tts_filepath, "wb") as out:
            out.write(response.audio_content)

        duration = get_mp3_duration(tts_filepath)

        logging.info(
            f"TTS audio generated with Google Cloud for segment {segment_index + 1} (Voice: {voice_name}, Speed: {speed}, Language: {language}).")
        return tts_filepath, duration

    except Exception as e:
        logging.error(f"Error generating TTS with Google Cloud for segment {segment_index + 1}: {e}")
        raise TTSError(f"Error generating TTS with Google Cloud: {e}")

def generate_tts_audio_for_segments(segments, temp_dir, target_language, sample_rate, channels):
    """Generates TTS audio for individual segments using the specified TTS engine."""
    logging.info(f"Starting TTS audio generation for segments to a temp dir using {TTS_ENGINE}...")
    tts_audio_files = []
    current_time = 0

    try:
        for i, segment in tqdm(enumerate(segments), total=len(segments), desc="Generating TTS for segments"):
            # Use the refined translated text
            translated_text = segment["translated_text"]

            # Handle empty segments
            if not translated_text.strip():
                logging.warning(f"Skipping segment {i + 1} because the translated text is empty.")
                continue

            if TTS_ENGINE == "openai":
                tts_filepath, duration = generate_tts_audio_with_openai(translated_text, i, temp_dir,
                                                                        TARGET_LANGUAGE, OPENAI_VOICE, OPENAI_SPEED)
            elif TTS_ENGINE == "google":
                tts_filepath, duration = generate_tts_audio_with_google_cloud(translated_text, i, temp_dir,
                                                                               TARGET_LANGUAGE, GOOGLE_CLOUD_VOICE,
                                                                               GOOGLE_CLOUD_SPEED)
            elif TTS_ENGINE == "gtts":
                tts_filename = f"tts_segment_{i}.mp3"
                tts_filepath = os.path.join(temp_dir, tts_filename)
                tts = gTTS(text=translated_text, lang=target_language, slow=False)
                tts.save(tts_filepath)
                duration = get_mp3_duration(tts_filepath)
            else:
                raise ValueError(f"Invalid TTS engine specified: {TTS_ENGINE}")

            # Update the segment with timing information
            segment["start"] = current_time
            segment["end"] = current_time + duration
            current_time += duration

            tts_audio_files.append(tts_filepath)

            logging.info(f"Finished generating TTS for segment {i + 1}.")

        # Merge TTS audio files into a single file
        merged_audio_path = os.path.join(temp_dir, "merged_tts_audio.mp3")
        logging.info(f"Merging TTS segments into a single audio file: {merged_audio_path}")
        merge_audio_chunks(tts_audio_files, merged_audio_path)

        logging.info(f"TTS audio generation completed. Merged audio saved to: '{merged_audio_path}'")
        return merged_audio_path, segments  # Return segments with updated timing

    except Exception as e:
        logging.error(f"Error during TTS audio generation: {e}")
        raise TTSError(f"Error during TTS audio generation: {e}")

def combine_audio_video(video_path, audio_path, output_path, timeout=FFMPEG_TIMEOUT):
    """Combines the video and new audio."""
    logging.info(f"Combining video and audio into '{output_path}'...")

    # Use absolute paths
    video_path = os.path.abspath(video_path)
    audio_path = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)

    try:
        # Simpler command for combining video and audio
        command = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",  # Re-encode audio to AAC (if needed)
            "-map", "0:v",  # Map the video stream from the input video
            "-map", "1:a",  # Map the audio stream from the new audio
            "-y",
            output_path
        ]

        # Execute the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=timeout)

        # Check for errors
        if process.returncode != 0:
            logging.error(f"ffmpeg exited with error code {process.returncode}:")
            logging.error(stderr.decode())
            raise FFmpegError(f"ffmpeg exited with error code {process.returncode} during video combination")

        logging.info(f"Successfully combined video and audio. Output saved to: {output_path}")

    except subprocess.TimeoutExpired:
        process.kill()
        logging.error(f"ffmpeg process timed out after {timeout} seconds during video combination.")
        raise FFmpegError(f"ffmpeg process timed out during video combination.")
    except Exception as e:
        logging.error(f"Error combining audio and video: {e}")
        raise FFmpegError(f"An error occurred during video combination: {e}")

# --- Main Processing Function ---

def process_video(video_path, output_path, srt_path, target_language, model, temp_dir):
    """Processes the video, performing all translation steps."""
    logging.info("Starting video processing...")
    os.makedirs(temp_dir, exist_ok=True)
    logging.info(f"Temporary files will be stored in '{temp_dir}'.")

    try:
        # 1. Extract audio in chunks
        audio_output_dir = os.path.join(temp_dir, "audio_chunks")
        os.makedirs(audio_output_dir, exist_ok=True)
        audio_chunks, total_duration = extract_audio_in_chunks(video_path, audio_output_dir)

        # 2. Merge audio chunks (if more than one)
        if len(audio_chunks) > 1:
            merged_audio_path = os.path.join(temp_dir, "merged_audio.mp3")
            merge_audio_chunks(audio_chunks, merged_audio_path)
        else:
            merged_audio_path = audio_chunks[0]

        # 3. Transcribe audio
        segments, total_words_original = transcribe_audio_in_chunks(audio_chunks, model, target_language)
        if not segments:
            raise Exception("Transcription failed.")

        # 4. Translate segments
        translated_segments = []
        total_words_translated = 0
        for segment in segments:
            translated_text = translate_text(segment["text"], target_language)
            translated_word_count = len(translated_text.split())
            total_words_translated += translated_word_count
            translated_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "word_count": segment["word_count"],
                "translated_text": translated_text,
                "translated_word_count": translated_word_count
            })

        logging.info(f"Total words after translation: {total_words_translated}")

        # 5. Refine all translated text with Gemini
        refined_segments = refine_text_with_gemini(translated_segments, target_language, total_words_original)

        # 6. Generate TTS audio
        sample_rate = 44100  # Common sample rate
        channels = 2  # Stereo
        translated_audio_path, updated_segments = generate_tts_audio_for_segments(
            refined_segments, temp_dir, target_language, sample_rate, channels
        )
        if not translated_audio_path:
            raise TTSError("TTS audio generation failed.")

        # 7. Create subtitle file (SRT) using updated segments
        if not create_subtitle_file(updated_segments, srt_path, target_language):
            raise SubtitleError("Subtitle creation failed.")

        # 8. Combine video and translated audio
        combine_audio_video(video_path, translated_audio_path, output_path)

        logging.info("Video processing completed.")
        return True

    except (AudioExtractionError, FFmpegError, TTSError, SubtitleError, GeminiError, Exception) as e:
        logging.error(f"Error during video processing: {e}")
        return False

    finally:
        # Clean up temporary files and directories
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logging.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logging.error(f"Error removing temporary directory {temp_dir}: {e}")

# --- Main Program ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video translator with TTS and subtitles.")
    parser.add_argument("--video", type=str, default=VIDEO_FILE, help="Path to the video file.")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_FILE, help="Path for the translated video.")
    parser.add_argument("--srt", type=str, default=SRT_FILE, help="Path for the output SRT file.")
    parser.add_argument("--language", type=str, default=TARGET_LANGUAGE, help="Target language for translation and TTS.")
    parser.add_argument("--model", type=str, default=MODEL_SIZE, help="Whisper model size (e.g., tiny, base, small, medium, large).")
    parser.add_argument("--temp_dir", type=str, default=TEMP_DIR, help="Directory for temporary files.")
    parser.add_argument("--audio_chunk_size", type=int, default=AUDIO_EXTRACTION_CHUNK_SIZE, help="Chunk size for audio extraction.")
    parser.add_argument("--transcribe_chunk_size", type=int, default=TRANSCRIBE_CHUNK_SIZE, help="Chunk size for transcription.")
    parser.add_argument("--tts_engine", type=str, default=TTS_ENGINE, help="TTS engine to use: 'openai', 'google', or 'gtts'.")
    parser.add_argument("--openai_voice", type=str, default=OPENAI_VOICE, help="Voice for OpenAI TTS.")
    parser.add_argument("--openai_speed", type=float, default=OPENAI_SPEED, help="Speed for OpenAI TTS.")
    parser.add_argument("--google_voice", type=str, default=GOOGLE_CLOUD_VOICE, help="Voice for Google Cloud TTS.")
    parser.add_argument("--google_speed", type=float, default=GOOGLE_CLOUD_SPEED, help="Speed for Google Cloud TTS.")
    args = parser.parse_args()

    # Validate TTS engine
    if args.tts_engine not in ["openai", "google", "gtts"]:
        logging.error(f"Invalid TTS engine: {args.tts_engine}. Choose from 'openai', 'google', or 'gtts'.")
        exit(1)

    TTS_ENGINE = args.tts_engine  # Update the global variable
    OPENAI_VOICE = args.openai_voice
    OPENAI_SPEED = args.openai_speed
    GOOGLE_CLOUD_VOICE = args.google_voice
    GOOGLE_CLOUD_SPEED = args.google_speed

    if not os.path.exists(args.video):
        logging.error(f"Error: Video file not found: {args.video}")
    else:
        model = WhisperModel(args.model, device=DEVICE)
        if process_video(
            args.video,
            args.output,
            args.srt,
            args.language,
            model,
            args.temp_dir,
        ):
            logging.info(f"The translated video has been saved to: {args.output}")
            logging.info(f"The SRT file has been saved to: {args.srt}")
        else:
            logging.error("Video translation with TTS failed.")