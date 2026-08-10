import os
import re
import shutil
import tempfile
import yt_dlp
from pydub import AudioSegment
import base64

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


# ============================================================
# DOWNLOAD DIRECTORY
# ============================================================

DOWNLOAD_DIR = "downloades"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ============================================================
# FFMPEG CONFIGURATION
# ============================================================
# Resolution order:
#   1. FFMPEG_DIR environment variable (optional manual override)
#   2. ffmpeg / ffprobe found on system PATH (works on Streamlit
#      Cloud via packages.txt, and locally once ffmpeg is on PATH)

FFMPEG_DIR = os.getenv("FFMPEG_DIR")

if FFMPEG_DIR:
    FFMPEG_PATH = os.path.join(FFMPEG_DIR, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe" if os.name == "nt" else "ffprobe")
else:
    FFMPEG_PATH = shutil.which("ffmpeg")
    FFPROBE_PATH = shutil.which("ffprobe")

# Check FFmpeg
if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(
        "FFmpeg not found. Install it and ensure it's on your system PATH, "
        "or set the FFMPEG_DIR environment variable to its folder."
    )

# Check FFprobe
if not FFPROBE_PATH or not os.path.exists(FFPROBE_PATH):
    raise FileNotFoundError(
        "FFprobe not found. Install it and ensure it's on your system PATH, "
        "or set the FFMPEG_DIR environment variable to its folder."
    )

# Tell pydub where FFmpeg is
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH


# ============================================================
# YOUTUBE COOKIES (optional, for bypassing 403 blocks)
# ============================================================
# On Streamlit Cloud, set a secret called YOUTUBE_COOKIES containing
# the full contents of a cookies.txt file exported from a logged-in
# YouTube session. Locally, this is optional — omit it and yt-dlp
# will just proceed without cookies.



COOKIE_FILE = None
_cookie_b64 = os.getenv("YOUTUBE_COOKIES_B64")

if _cookie_b64:
    _tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False)
    _tmp.write(base64.b64decode(_cookie_b64))
    _tmp.close()
    COOKIE_FILE = _tmp.name

print(f"DEBUG: cookie secret detected = {_cookie_b64 is not None}")
print(f"DEBUG: cookie file created = {COOKIE_FILE is not None}")
PROXY_URL = os.getenv("PROXY_URL")


# ============================================================
# YOUTUBE TRANSCRIPT (FAST PATH — avoids yt-dlp bot-check entirely)
# ============================================================
# Tries to fetch YouTube's own captions directly. This hits a totally
# different endpoint than yt-dlp's video download, so it is NOT subject
# to the "Sign in to confirm you're not a bot" block. Use this first;
# only fall back to download_youtube_audio() + Whisper if the video
# has no captions available.

def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from common URL formats
    (youtube.com/watch?v=..., youtu.be/..., etc.).
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL:\n{url}")


def get_youtube_transcript(url: str):
    """
    Try to fetch the YouTube caption transcript directly, without
    downloading audio or invoking yt-dlp at all.

    Returns:
        Full transcript text (str) if captions exist, otherwise None.
        Caller should fall back to download_youtube_audio() + Whisper
        when this returns None.
    """
    try:
        video_id = extract_video_id(url)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(segment["text"] for segment in transcript_list)

        if not text.strip():
            return None

        print(f"Transcript fetched via captions API ({len(text)} chars).")
        return text

    except (TranscriptsDisabled, NoTranscriptFound):
        print("No captions available for this video — will fall back to audio download.")
        return None

    except Exception as e:
        print(f"Transcript API failed unexpectedly: {e} — will fall back to audio download.")
        return None


# ============================================================
# DOWNLOAD YOUTUBE AUDIO
# ============================================================

def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL.

    Returns:
        Path to downloaded WAV file.
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "best",
        "outtmpl": output_path,
        "ffmpeg_location": os.path.dirname(FFMPEG_PATH),
        "extractor_args": {
            "youtube": {
                "player_client": ["web", "android", "tv"],
            }
        },
        "cookiefile": COOKIE_FILE,
        "proxy": PROXY_URL,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ],
        "noplaylist": True,
        "quiet": False,
        "nocheckcertificate": True,
    }

    try:
        print("Downloading YouTube audio...")
        print(f"URL: {url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            filename = ydl.prepare_filename(
                info
            )

            # Remove original extension
            filename_without_extension = os.path.splitext(
                filename
            )[0]

            # FFmpeg creates WAV
            wav_path = (
                filename_without_extension
                + ".wav"
            )

            if not os.path.exists(wav_path):
                raise FileNotFoundError(
                    f"WAV file was not created:\n{wav_path}"
                )

            print(
                f"YouTube audio downloaded:\n"
                f"{wav_path}"
            )

            return wav_path

    except yt_dlp.utils.DownloadError as e:

        raise RuntimeError(
            "Unable to download this YouTube video.\n\n"
            f"yt-dlp error:\n{str(e)}"
        )


# ============================================================
# CONVERT AUDIO / VIDEO TO WAV
# ============================================================

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to:

    - WAV
    - Mono
    - 16 kHz

    This format is suitable for Whisper.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"File not found:\n{input_path}"
        )

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    print(
        "Converting audio to "
        "16 kHz mono WAV..."
    )

    try:

        audio = AudioSegment.from_file(
            input_path
        )

        # Convert stereo -> mono
        audio = audio.set_channels(1)

        # Convert sample rate -> 16 kHz
        audio = audio.set_frame_rate(16000)

        # Export WAV
        audio.export(
            output_path,
            format="wav"
        )

    except Exception as e:

        raise RuntimeError(
            "Failed to convert audio to WAV.\n\n"
            f"Error: {str(e)}"
        )

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Converted WAV file was not created:\n"
            f"{output_path}"
        )

    print(
        f"Converted WAV:\n"
        f"{output_path}"
    )

    return output_path


# ============================================================
# SPLIT AUDIO INTO CHUNKS
# ============================================================

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:
    """
    Split WAV audio into chunks.

    Default:
        10 minutes per chunk
    """

    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"WAV file not found:\n{wav_path}"
        )

    print("Loading WAV audio...")

    audio = AudioSegment.from_wav(
        wav_path
    )

    # Make sure audio is Whisper-compatible
    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    # Check empty audio
    if len(audio) == 0:
        raise ValueError(
            "The audio file is empty."
        )

    chunk_ms = (
        chunk_minutes
        * 60
        * 1000
    )

    chunks = []

    for i, start in enumerate(
        range(
            0,
            len(audio),
            chunk_ms
        )
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        # Skip empty chunks
        if len(chunk) == 0:
            continue

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(
            chunk_path
        )

    if not chunks:
        raise ValueError(
            "No audio chunks were created."
        )

    print(
        f"Created {len(chunks)} "
        f"audio chunk(s)."
    )

    return chunks


# ============================================================
# PROCESS INPUT
# ============================================================

def process_input(source: str) -> list:
    """
    Process either:

    1. YouTube URL
    2. Local audio/video file

    Returns:
        List of 16 kHz mono WAV chunks.
    """

    if not source:
        raise ValueError(
            "No input source was provided."
        )

    source = source.strip()

    # --------------------------------------------------------
    # Handle Markdown URLs
    #
    # Example:
    #
    # [https://youtu.be/ABC](https://youtu.be/ABC)
    # --------------------------------------------------------

    if (
        source.startswith("[")
        and "](" in source
        and source.endswith(")")
    ):

        source = source.split(
            "](",
            1
        )[1][:-1].strip()

    # --------------------------------------------------------
    # YOUTUBE URL
    # --------------------------------------------------------

    if (
        source.startswith("http://")
        or source.startswith("https://")
    ):

        print(
            "Detected YouTube URL."
        )

        # Step 1:
        # Download YouTube audio
        downloaded_path = (
            download_youtube_audio(
                source
            )
        )

        # Step 2:
        # Convert to 16 kHz mono WAV
        wav_path = convert_to_wav(
            downloaded_path
        )

    # --------------------------------------------------------
    # LOCAL FILE
    # --------------------------------------------------------

    else:

        print(
            "Detected local audio/video file."
        )

        # Convert directly to
        # 16 kHz mono WAV
        wav_path = convert_to_wav(
            source
        )

    # --------------------------------------------------------
    # CHUNK AUDIO
    # --------------------------------------------------------

    print(
        "Chunking audio..."
    )

    chunks = chunk_audio(
        wav_path
    )

    print(
        f"Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks