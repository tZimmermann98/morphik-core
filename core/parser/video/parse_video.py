import base64
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

import assemblyai as aai
import cv2
import httpx
import litellm
import tomli

from core.config import get_settings
from core.models.video import ParseVideoResult, TimeSeriesData

logger = logging.getLogger(__name__)


def debug_object(title, obj):
    logger.debug("\n".join(["-" * 100, title, "-" * 100, f"{obj}", "-" * 100]))


class VoxtralTranscriber:
    """Transcribe video audio via an OpenAI-compatible /audio/transcriptions endpoint.

    Used instead of AssemblyAI where a self-hosted speech model is available
    (WWU serves Voxtral on gpt.uni-muenster.de), which avoids sending recordings
    to a third party.

    The endpoint returns flat text with no segment timestamps -- it rejects
    response_format=verbose_json -- so the audio is split into fixed windows and
    each window's start time becomes its timestamp. That is coarser than
    AssemblyAI's per-utterance timings but keeps TimeSeriesData meaningful for
    retrieval.
    """

    def __init__(self, model_config: Dict[str, Any], chunk_seconds: int = 120):
        self.api_base = (model_config.get("api_base") or "").rstrip("/")
        self.api_key = model_config.get("api_key")
        # registered_models carries a LiteLLM-style "hosted_vllm/<name>" id; the
        # HTTP endpoint wants the bare model name.
        self.model_name = (model_config.get("model_name") or "").split("/")[-1]
        self.chunk_seconds = max(int(chunk_seconds), 1)

        if not self.api_base or not self.model_name:
            raise ValueError("Voxtral transcription requires 'api_base' and 'model_name' in the registered model")

    def _extract_audio_chunk(self, video_path: str, start: float, out_path: str) -> bool:
        """Extract a mono 16 kHz MP3 window. Returns False when nothing was produced."""
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(self.chunk_seconds),
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            out_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out extracting audio at %.1fs", start)
            return False
        if result.returncode != 0:
            logger.warning("ffmpeg failed at %.1fs: %s", start, (result.stderr or "").strip()[:200])
            return False
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    def _transcribe_file(self, path: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path), fh, "audio/mpeg")}
            data = {"model": self.model_name, "response_format": "json"}
            resp = httpx.post(
                f"{self.api_base}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=httpx.Timeout(read=300.0, connect=30.0, write=120.0, pool=30.0),
            )
        resp.raise_for_status()
        payload = resp.json()
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        return (text or "").strip()

    def transcribe(self, video_path: str, duration: float) -> Dict[float, str]:
        """Transcribe the whole video, keyed by window start time in seconds."""
        if not duration or duration <= 0:
            logger.warning("Unknown video duration; skipping transcription")
            return {}

        time_to_text: Dict[float, str] = {}
        starts = [s for s in range(0, int(duration) + 1, self.chunk_seconds) if s < duration]
        logger.info(
            "Transcribing %.1fs of audio via %s in %d window(s) of %ds",
            duration,
            self.model_name,
            len(starts),
            self.chunk_seconds,
        )

        with tempfile.TemporaryDirectory() as workdir:
            for start in starts:
                chunk_path = os.path.join(workdir, f"chunk_{start}.mp3")
                if not self._extract_audio_chunk(video_path, float(start), chunk_path):
                    # A silent or video-only file yields no audio; that is not an error.
                    continue
                try:
                    text = self._transcribe_file(chunk_path)
                except Exception as e:
                    # One bad window must not lose the rest of the transcript.
                    logger.warning("Transcription failed for window at %ss: %s", start, str(e)[:200])
                    continue
                if text:
                    time_to_text[float(start)] = text

        logger.info("Transcription produced %d non-empty window(s)", len(time_to_text))
        return time_to_text


def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "../../../morphik.toml")
    with open(config_path, "rb") as f:
        return tomli.load(f)


class VisionModelClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["parser"]["vision"]
        self.model_key = self.config.get("model")
        self.settings = get_settings()

        # Get the model configuration from registered_models
        if not hasattr(self.settings, "REGISTERED_MODELS") or self.model_key not in self.settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{self.model_key}' not found in registered_models configuration")

        self.model_config = self.settings.REGISTERED_MODELS[self.model_key]
        logger.info(f"Initialized VisionModelClient with model_key={self.model_key}, config={self.model_config}")

        # Check if the model has vision capability
        if not self.model_config.get("vision", False):
            logger.warning(f"Model '{self.model_key}' does not have vision capability marked in config")

    async def get_frame_description(self, image_base64: str, context: str) -> str:
        # Create a system message
        system_message = {
            "role": "system",
            "content": "You are a video frame description assistant. Describe the frame clearly and concisely.",
        }

        # Determine if the model is OpenAI compatible or Ollama compatible based on model name
        model_name = self.model_config.get("model_name", "")

        if "ollama" in model_name.lower():
            # Ollama format with images parameter
            messages = [system_message, {"role": "user", "content": context}]

            model_params = {"model": model_name, "messages": messages, "images": [image_base64]}
        else:
            # Standard format with image_url (OpenAI, Anthropic, etc.)
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ]

            model_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 300,
            }

        # Add all additional parameters from the model config
        for key, value in self.model_config.items():
            if key not in ["model_name", "vision"]:
                model_params[key] = value

        # Use litellm for the completion
        response = await litellm.acompletion(**model_params)
        return response.choices[0].message.content


class VideoParser:
    def __init__(
        self,
        video_path: str,
        assemblyai_api_key: Optional[str] = None,
        frame_sample_rate: Optional[int] = None,
    ):
        """
        Initialize the video parser

        Args:
            video_path: Path to the video file
            assemblyai_api_key: Optional API key for AssemblyAI. If omitted, audio transcription is skipped.
            frame_sample_rate: Sample every nth frame for description (optional, defaults to config value)
        """
        logger.info(f"Initializing VideoParser for {video_path}")
        self.config = load_config()
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate or self.config["parser"]["vision"].get("frame_sample_rate", 120)
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

        self.transcript = TimeSeriesData(time_to_content={})

        # Select the transcription backend. [parser.transcription] provider may be
        # "voxtral" (self-hosted, OpenAI-compatible endpoint) or "assemblyai".
        # Defaults to assemblyai so upstream behaviour is unchanged when unset.
        self.transcriber = None
        self.voxtral = None
        transcription_cfg = self.config.get("parser", {}).get("transcription", {}) or {}
        provider = str(transcription_cfg.get("provider", "assemblyai")).lower()

        if provider == "voxtral":
            model_key = transcription_cfg.get("model")
            registered = getattr(get_settings(), "REGISTERED_MODELS", {}) or {}
            if model_key not in registered:
                logger.warning(
                    "Transcription provider is 'voxtral' but model '%s' is not in registered_models; "
                    "skipping transcription",
                    model_key,
                )
            else:
                try:
                    self.voxtral = VoxtralTranscriber(
                        registered[model_key],
                        chunk_seconds=transcription_cfg.get("chunk_seconds", 120),
                    )
                except Exception as e:
                    logger.warning("Could not initialize Voxtral transcription: %s", e)
        elif provider == "assemblyai" and assemblyai_api_key:
            aai.settings.api_key = assemblyai_api_key
            aai_config = aai.TranscriptionConfig(speaker_labels=True)
            self.transcriber = aai.Transcriber(config=aai_config)
        elif provider == "assemblyai":
            logger.warning("AssemblyAI API key is not available; skipping transcription")
        else:
            logger.warning("No transcription backend configured; skipping transcription")

        # Initialize vision model client
        self.vision_client = VisionModelClient(self.config)

        logger.info(f"Video loaded: {self.duration:.2f}s duration, {self.fps:.2f} FPS")

    def frame_to_base64(self, frame) -> str:
        """Convert a frame to base64 string"""
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("Failed to encode frame to JPEG")
            raise ValueError("Failed to encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def get_transcript_object(self) -> aai.Transcript:
        """
        Get the transcript object from AssemblyAI
        """
        if self.transcriber is None:
            raise ValueError("AssemblyAI API key is required for video transcription")

        logger.info("Starting video transcription")
        transcript = self.transcriber.transcribe(self.video_path)
        if transcript.status == "error":
            logger.error(f"Transcription failed: {transcript.error}")
            raise ValueError(f"Transcription failed: {transcript.error}")
        if not transcript.words:
            logger.warning("No words found in transcript")
        logger.info("Transcription completed successfully!")

        return transcript

    def get_transcript(self) -> TimeSeriesData:
        """
        Get timestamped transcript of the video using AssemblyAI

        Returns:
            TimeSeriesData object containing transcript
        """
        if self.voxtral is not None:
            time_to_text = self.voxtral.transcribe(self.video_path, self.duration)
            debug_object("Time to text", time_to_text)
            self.transcript = TimeSeriesData(time_to_content=time_to_text)
            return self.transcript

        if self.transcriber is None:
            self.transcript = TimeSeriesData(time_to_content={})
            return self.transcript

        logger.info("Starting video transcription")
        transcript = self.get_transcript_object()
        # divide by 1000 because assemblyai timestamps are in milliseconds
        time_to_text = {u.start / 1000: u.text for u in transcript.utterances} if transcript.utterances else {}
        debug_object("Time to text", time_to_text)
        self.transcript = TimeSeriesData(time_to_content=time_to_text)
        return self.transcript

    async def get_frame_descriptions(self) -> TimeSeriesData:
        """
        Get descriptions for sampled frames using configured vision model

        Returns:
            TimeSeriesData object containing frame descriptions
        """
        logger.info("Starting frame description generation")

        # Return empty TimeSeriesData if frame_sample_rate is -1 (captioning disabled)
        if self.frame_sample_rate == -1:
            logger.info("Frame captioning is disabled (frame_sample_rate = -1)")
            return TimeSeriesData(time_to_content={})

        frame_count = 0
        time_to_description = {}
        last_description = None
        logger.info("Starting main loop for frame description generation")
        while True:
            logger.info(f"Frame count: {frame_count}")
            ret, frame = self.cap.read()
            if not ret:
                logger.info("Reached end of video")
                break

            if frame_count % self.frame_sample_rate == 0:
                logger.info(f"Processing frame at {frame_count / self.fps:.2f}s")
                timestamp = frame_count / self.fps
                logger.debug(f"Processing frame at {timestamp:.2f}s")

                img_base64 = self.frame_to_base64(frame)

                if last_description:
                    previous_frame_context = last_description
                else:
                    previous_frame_context = "No previous frame description available, this is the first frame"

                description_instruction = (
                    "Describe this frame from a video. Focus on the main elements, actions, and any notable details."
                )
                previous_frame_section = (
                    "Here is a description of the previous frame:\n" "---\n" f"{previous_frame_context}\n" "---"
                )
                transcript_context = self.transcript.at_time(timestamp, padding=10)
                if transcript_context:
                    context = (
                        f"{description_instruction} Here is the transcript around the time of the frame:\n"
                        "---\n"
                        f"{transcript_context}\n"
                        "---\n\n"
                        f"{previous_frame_section}\n\n"
                        "In your response, only provide the description of the current frame, using the above "
                        "information as context."
                    )
                else:
                    context = (
                        f"{description_instruction}\n\n"
                        f"{previous_frame_section}\n\n"
                        "In your response, only provide the description of the current frame, using the above "
                        "information as context."
                    )

                last_description = await self.vision_client.get_frame_description(img_base64, context)
                time_to_description[timestamp] = last_description

            frame_count += 1

        logger.info(f"Generated descriptions for {len(time_to_description)} frames")
        return TimeSeriesData(time_to_content=time_to_description)

    async def process_video(self) -> ParseVideoResult:
        """
        Process the video to get frame descriptions and transcript when configured.

        Returns:
            Dictionary containing transcript and frame descriptions as TimeSeriesData objects
        """
        logger.info("Starting full video processing")
        metadata = {
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frame_sample_rate": self.frame_sample_rate,
        }
        # get_transcript() already returns an empty TimeSeriesData when no backend
        # is configured, so it is called unconditionally. Gating here on the
        # AssemblyAI transcriber specifically would skip the self-hosted backend.
        # Transcription runs first: frame captioning feeds nearby transcript text
        # to the vision model as context.
        result = ParseVideoResult(
            metadata=metadata,
            transcript=self.get_transcript(),
            frame_descriptions=await self.get_frame_descriptions(),
        )
        logger.info("Video processing completed successfully")
        return result

    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, "cap"):
            logger.debug("Releasing video capture resources")
            self.cap.release()
