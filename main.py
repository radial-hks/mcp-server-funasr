from fastmcp import FastMCP
from core.audio_processor import AudioProcessor
from core.speech_transcriber import SpeechTranscriber
from core.vad_processor import VADProcessor
from starlette.applications import Starlette
from starlette.routing import Mount

# Initialize FastMCP Server
mcp = FastMCP(
    title="FunASR Speech Services",
    description="MCP Server for speech validation, transcription, and voice activity detection using FunASR models.",
    version="0.1.0"
)

# Instantiate Tools
audio_processor = AudioProcessor()

# Configure SpeechTranscriber with specific defaults
# Using ModelScope IDs for clarity and to ensure models can be downloaded.
# Paraformer with VAD and Punctuation typically gives good, segmented results.
speech_transcriber_default_load_kwargs = {
    "vad_model": "./Model/speech_fsmn_vad_zh-cn-16k-common-pytorch", # FunASR's standard VAD
    # "punc_model": "Model/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", # FunASR's standard Punctuation
    # "vad_kwargs": {"max_single_segment_time": 30000} # Example if VAD model itself takes such load-time args
}
speech_transcriber_default_generate_kwargs = {
    "batch_size_s": 300, # Example: A common generate kwarg for Paraformer
    # "hotword": "默认热词" # Example: Default hotword if desired
}
speech_transcriber = SpeechTranscriber(
    default_model_name="./Model/SenseVoiceSmall", # Paraformer Large
    device="cpu", # Default to CPU, can be changed via load_model or task-specific calls if GPU is available
    ncpu=4,       # Default ncpu for CPU operations
    default_model_load_kwargs=speech_transcriber_default_load_kwargs,
    default_model_generate_kwargs=speech_transcriber_default_generate_kwargs
)

# Configure VADProcessor
vad_processor_default_load_kwargs = {
    # "max_single_segment_time": 30000 # Example if VAD model took such load-time args
}
vad_processor_default_generate_kwargs = {
    # VAD models often don't require many generate_kwargs for basic operation
}
vad_processor = VADProcessor(
    default_vad_model_name="./Model/speech_fsmn_vad_zh-cn-16k-common-pytorch", # FunASR's standard VAD
    device="cpu",
    ncpu=4,
    default_model_load_kwargs=vad_processor_default_load_kwargs,
    default_model_generate_kwargs=vad_processor_default_generate_kwargs
)

# Register AudioProcessor Tools
@mcp.tool(name="validate_audio_file", description="Validates an audio file to check if it's suitable for processing and provides its properties.")
def validate_audio_file(file_path: str) -> dict:
    """
    Checks if the audio file exists, is readable, and is a valid audio format.
    Returns a dictionary with validation status, message, and audio properties if valid.
    """
    return audio_processor.validate_audio(file_path=file_path)

# Register SpeechTranscriber Tools
@mcp.tool(name="start_speech_transcription", description="Starts an asynchronous speech transcription task for the given audio file. Allows specifying ASR model and generation parameters.")
def start_speech_transcription(audio_path: str, model_name: str = None, model_load_kwargs: dict = None, model_generate_kwargs: dict = None) -> dict:
    """
    Initiates a transcription task.
    Args:
        audio_path (str): Path to the audio file.
        model_name (str, optional): Specific ASR model to use for this task. Overrides the default.
        model_load_kwargs (dict, optional): Specific arguments for loading the ASR model (if model_name is specified and different).
        model_generate_kwargs (dict, optional): Specific arguments for the ASR model's generate method.
    Returns:
        dict: Contains task_id, status, and message.
    """
    # The SpeechTranscriber class itself handles None for model_name.
    # For kwargs, we default to empty dicts if None is passed by the client,
    # as the underlying SpeechTranscriber expects dicts (or handles None internally to use its defaults).
    effective_load_kwargs = model_load_kwargs if model_load_kwargs is not None else {}
    effective_generate_kwargs = model_generate_kwargs if model_generate_kwargs is not None else {}
    
    # Note: The SpeechTranscriber's start_transcription_task method will merge these
    # with its own defaults if these are empty or only partially specified.
    return speech_transcriber.start_transcription_task(
        audio_path=audio_path,
        model_name=model_name, 
        # model_load_kwargs are primarily used if model_name causes a model switch.
        # The current SpeechTranscriber.load_model uses its own default_model_load_kwargs if this is {}
        # This part is a bit tricky: `start_transcription_task` doesn't directly accept `model_load_kwargs`.
        # `load_model` is called internally if `model_name` changes.
        # For now, we pass them, but `SpeechTranscriber` would need adjustment if direct load_kwargs override during task start is desired for a model switch.
        # The current `SpeechTranscriber.start_transcription_task` does not use `model_load_kwargs` in its signature.
        # This is a slight mismatch from the tool registration example.
        # Let's adjust the tool to match the transcriber's current capability.
        # The transcriber's `load_model` takes `model_load_kwargs`.
        # The `start_transcription_task` only takes `model_generate_kwargs`.
        # If a model switch happens in `start_transcription_task`, it uses existing `self.current_model_load_kwargs` or `self.default_model_load_kwargs`.
        # For simplicity, we'll remove model_load_kwargs from this tool's direct signature for now,
        # as it's not directly plumbed into start_transcription_task.
        # Re-adding model_load_kwargs for explicit model loading tool later would be cleaner.
        model_generate_kwargs=effective_generate_kwargs
    )

@mcp.tool(name="get_transcription_task_status", description="Queries the status of a previously started speech transcription task.")
def get_transcription_task_status(task_id: str) -> dict:
    """
    Retrieves the current status and metadata of a transcription task.
    Args:
        task_id (str): The unique ID of the transcription task.
    Returns:
        dict: Task status information.
    """
    return speech_transcriber.get_task_status(task_id=task_id)

@mcp.tool(name="get_transcription_result", description="Retrieves the result of a completed speech transcription task.")
def get_transcription_result(task_id: str) -> dict:
    """
    Fetches the transcription result for a specific task ID.
    Args:
        task_id (str): The unique ID of the transcription task.
    Returns:
        dict: Transcription result if completed, or status information otherwise.
    """
    return speech_transcriber.get_transcription_result(task_id=task_id)

@mcp.tool(name="load_asr_model", description="Loads or reloads a specific ASR model, making it the default for subsequent tasks unless overridden. Returns status of operation.")
def load_asr_model(model_name: str, device: str = None, model_load_kwargs: dict = None) -> dict:
    """
    Explicitly loads an ASR model.
    Args:
        model_name (str): The FunASR model identifier to load.
        device (str, optional): Device to load the model on (e.g., "cpu", "cuda:0"). Uses instance default if None.
        model_load_kwargs (dict, optional): Specific arguments for loading the ASR model.
    Returns:
        dict: Status of the model loading operation.
    """
    effective_load_kwargs = model_load_kwargs if model_load_kwargs is not None else {}
    return speech_transcriber.load_model(model_name=model_name, device=device, model_load_kwargs=effective_load_kwargs)


# Register VADProcessor Tools
@mcp.tool(name="get_voice_activity_segments", description="Detects speech segments in an audio file using a Voice Activity Detection (VAD) model.")
def get_voice_activity_segments(audio_path: str, vad_model_name: str = None, model_load_kwargs: dict = None, model_generate_kwargs: dict = None) -> dict:
    """
    Identifies speech segments in an audio file.
    Args:
        audio_path (str): Path to the audio file.
        vad_model_name (str, optional): Specific VAD model to use. Overrides the default.
        model_load_kwargs (dict, optional): Specific arguments for loading the VAD model (if vad_model_name is specified and different).
        model_generate_kwargs (dict, optional): Specific arguments for the VAD model's generate method.
    Returns:
        dict: Contains status, list of speech segments (start_ms, end_ms), and other info.
    """
    effective_load_kwargs = model_load_kwargs if model_load_kwargs is not None else {}
    effective_generate_kwargs = model_generate_kwargs if model_generate_kwargs is not None else {}
    
    return vad_processor.get_speech_segments(
        audio_path=audio_path,
        vad_model_name=vad_model_name,
        model_load_kwargs=effective_load_kwargs,
        model_generate_kwargs=effective_generate_kwargs
    )

@mcp.tool(name="load_vad_model", description="Loads or reloads a specific VAD model, making it the default for subsequent VAD tasks unless overridden. Returns status of operation.")
def load_vad_model(model_name: str, device: str = None, ncpu: int = None, model_load_kwargs: dict = None) -> dict:
    """
    Explicitly loads a VAD model.
    Args:
        model_name (str): The FunASR VAD model identifier to load.
        device (str, optional): Device to load the model on (e.g., "cpu", "cuda:0"). Uses instance default if None.
        ncpu (int, optional): Number of CPU threads if device is CPU. Uses instance default if None.
        model_load_kwargs (dict, optional): Specific arguments for loading the VAD model.
    Returns:
        dict: Status of the model loading operation.
    """
    effective_load_kwargs = model_load_kwargs if model_load_kwargs is not None else {}
    # The VADProcessor.load_vad_model also takes ncpu directly, let's ensure it's passed if provided.
    return vad_processor.load_vad_model(
        model_name=model_name, 
        device=device, 
        ncpu=ncpu, 
        model_load_kwargs=effective_load_kwargs
    )

app = Starlette(
    routes=[
        Mount('/', app=mcp.sse_app()),
    ]
)

# Running the Server
if __name__ == "__main__":
    print("Starting FunASR MCP Server...")
    # To run with customized host/port or other uvicorn settings,
    # you might use `uvicorn.run(mcp.app, host="0.0.0.0", port=8000, ...)`
    # or `fastmcp.server.run_server(mcp, host="0.0.0.0", port=8000)`
    # For simplicity, mcp.run() uses uvicorn with default settings.
    mcp.run(host="0.0.0.0", port=8000)


# uvicorn main:app --host 0.0.0.0 --port 9000
# http://0.0.0.0:9000/sse