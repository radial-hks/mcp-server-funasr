# FunASR-Powered MCP Server (MCPServer)

## Overview

MCPServer is a Python-based server that leverages Alibaba's FunASR library to provide speech processing services through the FastMCP framework. It offers tools for:

- **Audio Validation:** Checking if audio files are valid and readable, and providing their properties.
- **Speech Transcription:** Asynchronously transcribing speech from audio files using advanced ASR models like Paraformer. Supports managing transcription tasks and retrieving results, including detailed timestamp information.
- **Voice Activity Detection (VAD):** Identifying speech segments in audio files.

The server is designed to be extensible and allows for dynamic loading and switching of ASR and VAD models.

## Features

-   **Audio File Validation:** Verifies audio file integrity, readability, and format.
-   **Asynchronous Speech-to-Text Transcription:** Non-blocking transcription suitable for long audio files.
-   **Transcription Task Management:** Start tasks, query status, and retrieve results.
-   **Detailed Transcription Results:** Access to full transcription text, segment-level start/end times, and word-level timestamps (if provided by the ASR model).
-   **Voice Activity Detection (VAD):** Returns precise start and end timestamps of speech segments in an audio file.
-   **Multi-Model Support:** Leverages FunASR's diverse model zoo for both ASR and VAD.
-   **Dynamic Model Configuration:**
    -   Specify models per transcription or VAD request.
    -   Explicitly load/switch the default ASR and VAD models used by the server instance.
-   **Configurable Model Parameters:** Pass specific loading and generation arguments to FunASR models.

## Prerequisites

-   **Python:** 3.8+
-   **Pip:** For installing Python packages.
-   **MODELSCOPE_API_TOKEN (Optional):**
    -   FunASR downloads models from ModelScope. If you encounter rate limits or need to access private models, you might need to set the `MODELSCOPE_API_TOKEN` environment variable.
    -   You can obtain a token from the [ModelScope website](https://www.modelscope.cn/).
    -   Set it in your environment: `export MODELSCOPE_API_TOKEN="YOUR_TOKEN_HERE"`

## Setup and Installation

1.  **Clone the Repository (if applicable):**
    If this server is part of a larger repository, clone it. Otherwise, ensure you have the `MCPServer` directory and its contents.

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Navigate to the `MCPServer` directory (the one containing this README and `server.py`).
    Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `fastmcp`, `funasr`, and their dependencies, including PyTorch (CPU version by default if not specified otherwise by FunASR's direct dependencies). If you have specific PyTorch needs (e.g., a GPU version), it's recommended to install PyTorch manually before running the command above, following instructions from the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Running the Server

1.  **Navigate to the `MCPServer` directory.**
2.  **Run the server application:**
    ```bash
    python server.py
    ```
3.  The server will start, and you should see log output indicating it's running, typically on `http://0.0.0.0:8000`. On the first run, FunASR will download the default ASR and VAD models, which may take some time.

## Available MCP Tools

You can interact with these tools using any MCP client (e.g., `mcp_client` or via HTTP requests). The server provides the following tools:

---

### 1. `validate_audio_file`
   - **Description:** Validates an audio file to check if it's suitable for processing and provides its properties.
   - **Parameters:**
     - `file_path` (str, required): Path to the audio file.
   - **Example Return (Success):**
     ```json
     {
       "status": "valid",
       "message": "Audio file is valid.",
       "details": {
         "samplerate": 16000,
         "channels": 1,
         "duration": 10.5,
         "formatted_duration": "00:10.500",
         "format": "WAV",
         "subtype": "PCM_16"
       }
     }
     ```
   - **Example Return (Error - File Not Found):**
     ```json
     {
        "status": "invalid",
        "message": "Error: File not found at 'path/to/non_existent_audio.wav'.",
        "details": null
     }
     ```

---

### 2. `start_speech_transcription`
   - **Description:** Starts an asynchronous speech transcription task for the given audio file. Allows specifying ASR model and generation parameters.
   - **Parameters:**
     - `audio_path` (str, required): Path to the audio file.
     - `model_name` (str, optional): Specific ASR model to use for this task (e.g., a ModelScope ID). Overrides the server's current default ASR model. If the specified model is not already loaded with compatible settings, the server will attempt to load it using its default load parameters for that model or the instance's general default load parameters.
     - `model_generate_kwargs` (dict, optional): Specific arguments for the ASR model's `generate` method (e.g., `{"batch_size_s": 60, "hotword": "特定热词"}`). These override any default generation arguments set in the server for the current ASR model.
   - **Example Return (Success):**
     ```json
     {
       "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
       "status": "processing_started",
       "message": "Transcription task started and is now processing."
     }
     ```
   - **Example Return (Error - Invalid Audio):**
     ```json
     {
        "task_id": null,
        "status": "error",
        "message": "Error: File at '/path/to/your/bad_audio.wav' is not a valid audio file or is corrupted. Details: <error from soundfile>",
        "details": null
     }
     ```
   - **Example Return (Error - Model Load Failure during task):**
     ```json
     {
        "task_id": null,
        "status": "error",
        "message": "Failed to switch to model 'non_existent_model_id'. Error: Error loading model 'non_existent_model_id': <actual error>"
     }
     ```

---

### 3. `get_transcription_task_status`
   - **Description:** Queries the status of a previously started speech transcription task.
   - **Parameters:**
     - `task_id` (str, required): The unique ID of the transcription task.
   - **Example Return (Processing):**
     ```json
     {
        "status": "processing",
        "audio_path": "/path/to/your/audio.wav",
        "model_used": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "submitted_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00",
        "details_from_validation": { /* ... audio details from validate_audio ... */ },
        "model_generate_kwargs": {"batch_size_s": 300, "hotword": "魔搭"},
        "processing_started_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
     }
     ```
    - **Example Return (Completed):**
     ```json
     {
        "status": "completed",
        "audio_path": "/path/to/your/audio.wav",
        "model_used": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "submitted_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00",
        "details_from_validation": { /* ... */ },
        "model_generate_kwargs": { /* ... */ },
        "processing_started_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00",
        "result": [ /* ... actual transcription result ... */ ],
        "completed_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
     }
     ```
    - **Example Return (Error - Task Not Found):**
      ```json
      {
          "status": "error",
          "message": "Task ID not found."
      }
      ```

---

### 4. `get_transcription_result`
   - **Description:** Retrieves the result of a completed speech transcription task.
   - **Parameters:**
     - `task_id` (str, required): The unique ID of the transcription task.
   - **Example Return (Success/Completed):**
     ```json
     {
       "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
       "status": "completed",
       "result": [
         {
           "text": "这是 一段 测试 文本",
           "start": 120,
           "end": 2850,
           "timestamp": [[120, 300], [330, 500], [550, 900], [920, 1200]]
         }
       ],
       "completed_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
     }
     ```
   - **Example Return (Task Still Processing):**
     ```json
     {
        "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "status": "processing",
        "message": "Transcription not yet completed or has failed."
     }
     ```
   - **Example Return (Task Failed):**
     ```json
     {
        "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "status": "failed",
        "message": "Transcription failed.",
        "error_details": "Description of the error during transcription.",
        "failed_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
     }
     ```

---

### 5. `load_asr_model`
   - **Description:** Loads or reloads a specific ASR model, making it the default for subsequent tasks unless overridden. Returns status of operation.
   - **Parameters:**
     - `model_name` (str, required): The FunASR model identifier to load (e.g., a ModelScope ID).
     - `device` (str, optional): Device to load the model on (e.g., "cpu", "cuda:0"). Uses instance default if None.
     - `model_load_kwargs` (dict, optional): Specific arguments for loading the ASR model (e.g., `{"ncpu": 2, "vad_model": "other-vad-id", "punc_model": "other-punc-id"}`). These will be passed to `funasr.AutoModel`.
   - **Example Return (Success):**
     ```json
     {
       "status": "success",
       "message": "Model 'iic/speech_paraformer-large-en-16k-common-vocab10020' loaded successfully on cpu with load_kwargs: {'ncpu': 2, 'vad_model': 'fsmn-vad'}."
     }
     ```
   - **Example Return (Error):**
     ```json
     {
        "status": "error",
        "message": "Error loading model 'invalid-model-id': <FunASR or ModelScope error details>"
     }
     ```

---

### 6. `get_voice_activity_segments`
   - **Description:** Detects speech segments in an audio file using a Voice Activity Detection (VAD) model.
   - **Parameters:**
     - `audio_path` (str, required): Path to the audio file.
     - `vad_model_name` (str, optional): Specific VAD model to use. Overrides the server's current default VAD model.
     - `model_load_kwargs` (dict, optional): Specific arguments for loading the VAD model if `vad_model_name` is specified and different from the currently loaded one.
     - `model_generate_kwargs` (dict, optional): Specific arguments for the VAD model's `generate` method.
   - **Example Return (Success):**
     ```json
     {
       "status": "success",
       "segments": [ [100, 2500], [3000, 5500] ],
       "audio_path": "path/to/your/audio.wav",
       "vad_model_used": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
       "generate_kwargs_used": {},
       "audio_details": { /* ... audio properties ... */ }
     }
     ```
    - **Example Return (Error - VAD Processing Failed):**
     ```json
     {
        "status": "error",
        "message": "VAD processing failed for 'path/to/audio.wav': <FunASR error details>",
        "audio_path": "path/to/audio.wav",
        "vad_model_used": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
     }
     ```

---

### 7. `load_vad_model`
   - **Description:** Loads or reloads a specific VAD model, making it the default for subsequent VAD tasks unless overridden. Returns status of operation.
   - **Parameters:**
     - `model_name` (str, required): The FunASR VAD model identifier to load.
     - `device` (str, optional): Device to load the model on. Uses instance default if None.
     - `ncpu` (int, optional): Number of CPU threads if device is CPU. Uses instance default if None.
     - `model_load_kwargs` (dict, optional): Specific arguments for loading the VAD model.
   - **Example Return (Success):**
     ```json
     {
       "status": "success",
       "message": "VAD Model 'damo/speech_fsmn_vad_zh-cn-16k-common-pytorch' loaded successfully on cpu with load_kwargs: {'ncpu': 2}."
     }
     ```

---

## Example Usage with `curl`

Here's how you might call the `start_speech_transcription` tool using `curl`:

```bash
# Ensure the audio_path is accessible by the server.
# For paths with spaces or special characters, ensure proper JSON escaping if needed.
curl -X POST http://localhost:8000/mcp/start_speech_transcription \
     -H "Content-Type: application/json" \
     -d '{"params": {"audio_path": "/path/to/your/audio_sample.wav"}}'
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": "some_client_generated_id",
  "result": {
    "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "status": "processing_started",
    "message": "Transcription task started and is now processing."
  }
}
```

To get the result later, use the `task_id` from the response:
```bash
curl -X POST http://localhost:8000/mcp/get_transcription_result \
     -H "Content-Type: application/json" \
     -d '{"params": {"task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}}'
```

## Model Configuration

-   **Default Models:** The server initializes with default ASR and VAD models specified in `MCPServer/server.py`.
    -   Default ASR: `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` (pre-configured with VAD `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` and Punctuation `damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`).
    -   Default VAD: `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch`.
-   **Per-Request Model Specification:**
    -   For `start_speech_transcription`, use the `model_name` parameter to specify a different ASR model for that specific task.
    -   For `get_voice_activity_segments`, use the `vad_model_name` parameter for a task-specific VAD model.
    -   If a model specified per-request is different from the currently loaded one, the server will attempt to load it. This new model (if loaded successfully) becomes the current default for that processor type (ASR or VAD) for subsequent requests that *don't* specify a model.
-   **Global Model Loading:**
    -   Use the `load_asr_model` tool to change the default ASR model for the `SpeechTranscriber` instance.
    -   Use the `load_vad_model` tool to change the default VAD model for the `VADProcessor` instance.
    -   These tools also allow specifying `device`, `ncpu` (for VAD), and `model_load_kwargs` for finer control over model loading (e.g., for ASR, `model_load_kwargs` can include VAD and Punctuation model IDs).
-   **Model Arguments:**
    -   `model_load_kwargs`: Can be passed to `load_asr_model` and `load_vad_model` to control how models are loaded (e.g., to specify sub-models like VAD/Punctuation for an ASR pipeline, or other model-specific loading parameters like `max_single_segment_time` if supported by the VAD model at load time).
    -   `model_generate_kwargs`: Can be passed to `start_speech_transcription` and `get_voice_activity_segments` to control the behavior of the model's inference/generation step (e.g., `batch_size_s`, `hotword` for ASR; VAD models usually have fewer generation-time parameters).

## Troubleshooting

-   **Model Download Issues:**
    -   Ensure the server has internet access to download models from ModelScope (hub.modelscope.cn).
    -   If you encounter persistent download errors or authentication issues (e.g., HTTP 401/403), set the `MODELSCOPE_API_TOKEN` environment variable.
    -   Check available disk space in the ModelScope cache directory (usually `~/.cache/modelscope/hub/`).
-   **Dependency Conflicts:**
    -   Using a Python virtual environment is highly recommended to avoid conflicts with system-wide packages or other projects.
-   **PyTorch Version:**
    -   FunASR requires a specific range of PyTorch versions. If `requirements.txt` doesn't fetch a compatible version, or if you have an existing conflicting PyTorch installation, manual installation of a compatible PyTorch version might be needed. Refer to FunASR's documentation for compatible PyTorch versions.
-   **Resource Limits:**
    -   ASR models, especially larger ones like Paraformer, can be memory and CPU intensive. Ensure the server has adequate resources. For CPU inference, performance is heavily tied to `ncpu` (number of CPU threads, configurable in `SpeechTranscriber` and `VADProcessor`) and the CPU's capabilities.
-   **File Paths:**
    -   Ensure that `audio_path` provided to the tools is an absolute path or a path relative to where the `server.py` script is run, and that it is accessible by the server process.
-   **CUDA/GPU Issues (if using GPU):**
    -   If using GPU (`device="cuda:X"`), ensure NVIDIA drivers, CUDA Toolkit, and a GPU-enabled version of PyTorch are correctly installed and compatible. Use tools like `nvidia-smi` and `torch.cuda.is_available()` for diagnostics.
```
