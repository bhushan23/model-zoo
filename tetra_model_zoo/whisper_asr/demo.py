import whisper

from tetra_model_zoo.utils.asset_loaders import maybe_download_s3_data
from tetra_model_zoo.whisper_asr.app import WhisperApp, load_audio, load_mel_filter
from tetra_model_zoo.whisper_asr.model import MODEL_NAME, Whisper

if __name__ == "__main__":
    # For other model sizes, see https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17
    model = whisper.load_model("tiny.en")
    app = WhisperApp(Whisper.from_pretrained())

    # Load audio into mel spectrogram
    mel_filter_path = maybe_download_s3_data(
        "whisper/openai_assets/mel_filters.npz",
        MODEL_NAME,
    )
    mel_filter = load_mel_filter(mel_filter_path)

    audio_path = maybe_download_s3_data(
        "whisper/audio/jfk.npz",
        MODEL_NAME,
    )
    mel_input = load_audio(mel_filter, audio_path)

    # Perform transcription
    transcription = app.transcribe(mel_input)
    print("Transcription:", transcription)
