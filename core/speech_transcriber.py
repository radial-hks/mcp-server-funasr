import funasr
import uuid
import os
import threading
from datetime import datetime, timezone # Updated import
import time # For example usage
from .audio_processor import AudioProcessor

# if "MODELSCOPE_API_TOKEN" not in os.environ:
#     print("Note: MODELSCOPE_API_TOKEN environment variable not set. Some models might require it for download.")

class SpeechTranscriber:
    def __init__(self, default_model_name: str = "paraformer-zh", 
                device: str = "cpu", 
                ncpu: int = 4, 
                default_model_load_kwargs: dict = None,
                default_model_generate_kwargs: dict = None):
        """
        Constructor for SpeechTranscriber.

        Args:
            default_model_name (str): Default FunASR model to load. 模型名称或者是模型路径。
            device (str): Device to run the model on (e.g., "cpu", "cuda:0").
            ncpu (int): Number of CPU threads for CPU inference (used if device is "cpu").
            default_model_load_kwargs (dict, optional): Default kwargs for model loading (funasr.AutoModel).
                                                    'ncpu' will be added/overridden if device is 'cpu'.
            default_model_generate_kwargs (dict, optional): Default kwargs for model.generate().
        """
        self.default_model_name = default_model_name
        self.device = device
        self.ncpu = ncpu # Primarily for CPU device context
        
        self.default_model_load_kwargs = default_model_load_kwargs or {}
        if self.device == "cpu" and 'ncpu' not in self.default_model_load_kwargs:
            self.default_model_load_kwargs['ncpu'] = self.ncpu
        elif self.device == "cpu" and self.default_model_load_kwargs.get('ncpu') != self.ncpu:
            print(f"Warning: 'ncpu' in default_model_load_kwargs ({self.default_model_load_kwargs.get('ncpu')}) "
                f"differs from ncpu parameter ({self.ncpu}). Using value from default_model_load_kwargs.")
            self.ncpu = self.default_model_load_kwargs.get('ncpu')


        self.default_model_generate_kwargs = default_model_generate_kwargs or {}

        self.current_model_name = None
        self.model = None
        self.current_model_load_kwargs = None # Kwargs used for loading current model
        self.audio_processor = AudioProcessor()

        self.tasks = {}
        self.tasks_lock = threading.Lock()

        # Initial model load
        load_status = self.load_model(
            model_name=self.default_model_name, 
            device=self.device, 
            model_load_kwargs=self.default_model_load_kwargs
        )
        if load_status["status"] == "error":
            print(f"FATAL: Initial model load failed for {self.default_model_name}. Error: {load_status['message']}")

    def load_model(self, model_name: str, device: str = None, model_load_kwargs: dict = None) -> dict:
        """
        Loads or reloads an ASR model.

        Args:
            model_name (str): The FunASR model to load.
            device (str, optional): Device for the model. Defaults to instance's current device.
            model_load_kwargs (dict, optional): Kwargs for funasr.AutoModel. Defaults to instance's default_model_load_kwargs.
                                            If device is 'cpu', 'ncpu' will be managed.

        Returns:
            dict: Status of the model loading operation.
        """
        effective_device = device if device is not None else self.device
        # Prioritize model_load_kwargs from args, then instance default, then empty dict
        effective_model_load_kwargs = model_load_kwargs if model_load_kwargs is not None else self.default_model_load_kwargs.copy()


        if effective_device == "cpu":
            # Ensure ncpu is correctly set in kwargs for CPU devices
            # If ncpu is explicitly in effective_model_load_kwargs, it takes precedence.
            # Otherwise, use self.ncpu (which might have been updated from initial default_model_load_kwargs).
            if 'ncpu' not in effective_model_load_kwargs:
                effective_model_load_kwargs['ncpu'] = self.ncpu
        else: # For non-CPU devices, remove ncpu if it's there
            if 'ncpu' in effective_model_load_kwargs:
                del effective_model_load_kwargs['ncpu']
        
        try:
            self.model = funasr.AutoModel(model=model_name, device=effective_device, **effective_model_load_kwargs)
            self.current_model_name = model_name
            self.device = effective_device # Update instance device
            self.current_model_load_kwargs = effective_model_load_kwargs # Store the actual kwargs used
            if effective_device == "cpu" and 'ncpu' in effective_model_load_kwargs:
                self.ncpu = effective_model_load_kwargs['ncpu'] # Update instance ncpu if specified for CPU

            return {"status": "success", "message": f"Model '{model_name}' loaded successfully on {self.device} with load_kwargs: {effective_model_load_kwargs}."}
        except Exception as e:
            error_message = f"Error loading model '{model_name}': {e}"
            print(error_message)
            return {"status": "error", "message": error_message}

    def _transcribe(self, task_id: str, audio_path: str, model_generate_kwargs: dict):
        """
        Private method to perform transcription in a separate thread.
        """
        try:
            with self.tasks_lock:
                self.tasks[task_id]["status"] = "processing"
                self.tasks[task_id]["processing_started_at"] = datetime.now(timezone.utc).isoformat()

            # Perform transcription
            # Ensure self.model is valid before calling generate
            if not self.model:
                raise RuntimeError(f"No model loaded for task {task_id}. Current model: {self.current_model_name}")

            transcription_result = self.model.generate(input=audio_path, **model_generate_kwargs)
            
            with self.tasks_lock:
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["result"] = transcription_result # This stores the rich output from FunASR
                self.tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            error_message = f"Transcription failed for task {task_id}: {e}"
            print(error_message) # Log this error
            with self.tasks_lock:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error_message"] = str(e)
                self.tasks[task_id]["failed_at"] = datetime.now(timezone.utc).isoformat()

    def start_transcription_task(self, audio_path: str, model_name: str = None, model_generate_kwargs: dict = None) -> dict:
        """
        Starts a transcription task asynchronously.
        Validates audio, loads model if necessary, and generates a task ID.

        Args:
            audio_path (str): Path to the audio file.
            model_name (str, optional): Specific model to use. Defaults to the current model.
            model_generate_kwargs (dict, optional): Kwargs for model.generate(). Overrides default.

        Returns:
            dict: Information about the started task or an error.
        """
        validation_result = self.audio_processor.validate_audio(audio_path)
        if validation_result["status"] == "invalid":
            return {"task_id": None, "status": "error", "message": validation_result["message"], "details": validation_result.get("details")}

        # Determine model to use and load if necessary
        model_to_use = self.current_model_name
        if model_name and model_name != self.current_model_name:
            print(f"Switching model from '{self.current_model_name}' to '{model_name}' for this task.")
            load_result = self.load_model(model_name, device=self.device, model_load_kwargs=self.current_model_load_kwargs)
            if load_result["status"] == "error":
                return {"task_id": None, "status": "error", "message": f"Failed to switch to model '{model_name}'. Error: {load_result['message']}"}
            model_to_use = model_name
        
        if not self.model: 
            return {"task_id": None, "status": "error", "message": "No ASR model is loaded. Cannot start transcription."}

        task_id_str = str(uuid.uuid4())
        
        final_model_generate_kwargs = self.default_model_generate_kwargs.copy()
        if model_generate_kwargs is not None:
            final_model_generate_kwargs.update(model_generate_kwargs)

        with self.tasks_lock:
            self.tasks[task_id_str] = {
                "status": "pending",
                "audio_path": audio_path,
                "model_used": model_to_use,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "details_from_validation": validation_result.get("details"),
                "model_generate_kwargs": final_model_generate_kwargs, 
            }
        
        try:
            thread = threading.Thread(target=self._transcribe, args=(task_id_str, audio_path, final_model_generate_kwargs))
            thread.start()
        except Exception as e:
            with self.tasks_lock:
                self.tasks[task_id_str]["status"] = "failed"
                self.tasks[task_id_str]["error_message"] = f"Failed to start transcription thread: {e}"
                self.tasks[task_id_str]["failed_at"] = datetime.now(timezone.utc).isoformat()
            return {"task_id": task_id_str, "status": "error", "message": f"Failed to start transcription thread: {e}"}

        return {
            "task_id": task_id_str,
            "status": "processing_started",
            "message": "Transcription task started and is now processing.",
        }

    def get_task_status(self, task_id: str) -> dict:
        with self.tasks_lock:
            task_info = self.tasks.get(task_id)
            if task_info:
                return task_info.copy() 
            else:
                return {"status": "error", "message": "Task ID not found."}

    def get_transcription_result(self, task_id: str) -> dict:
        with self.tasks_lock:
            task_info = self.tasks.get(task_id)
            if not task_info:
                return {"status": "error", "message": "Task ID not found."}
            
            current_status = task_info.get("status")
            if current_status == "completed":
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task_info.get("result"),
                    "completed_at": task_info.get("completed_at")
                }
            elif current_status == "failed":
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "message": "Transcription failed.",
                    "error_details": task_info.get("error_message"),
                    "failed_at": task_info.get("failed_at")
                }
            else: 
                return {
                    "task_id": task_id,
                    "status": current_status,
                    "message": "Transcription not yet completed or has failed."
                }

if __name__ == '__main__':
    print("Attempting to initialize SpeechTranscriber with default model...")
    
    # Ensures VAD and Punctuation are part of the ASR pipeline for richer output.
    # This is important for getting segment-level timestamps and potentially word-level ones.
    default_load_kwargs = {
        "vad_model": "../Model/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
        #TODO： "punc_model": "ct-punc", 
        # "vad_kwargs": {"max_single_segment_time": 30000} # Example VAD specific config if needed
    }
    # Example generate_kwargs, 'hotword' is common.
    default_generate_kwargs = {"batch_size_s": 300, "hotword": "魔搭"} 

    transcriber = SpeechTranscriber(
        default_model_name="../Model/SenseVoiceSmall",
        device="cpu", 
        ncpu=2, 
        default_model_load_kwargs=default_load_kwargs,
        default_model_generate_kwargs=default_generate_kwargs
    )
    
    if not transcriber.model:
        print("SpeechTranscriber initialization failed to load the default model. Exiting due to model load failure.")
        exit()
    print(f"SpeechTranscriber initialized. Current model: {transcriber.current_model_name}")

    dummy_audio_path = "E:\Code\PythonDir\MCP\mcp-server-funasr\Data\_20240821153822.mp3"
    if not os.path.exists(dummy_audio_path):
        try:
            import soundfile as sf
            import numpy as np
            samplerate = 16000
            duration = 3 # seconds, to get a few words
            frequency1 = 440 # A4
            frequency2 = 659 # E5
            
            t = np.linspace(0, duration, int(samplerate * duration), False, dtype=np.float32)
            # Create a sound with two varying frequencies to simulate speech-like audio
            data = 0.3 * np.sin(2 * np.pi * frequency1 * t * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))) # Modulate freq1
            data += 0.2 * np.sin(2 * np.pi * frequency2 * t * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) # Modulate freq2
            data /= np.max(np.abs(data)) # Normalize
            
            sf.write(dummy_audio_path, data, samplerate, format='WAV', subtype='PCM_16')
            print(f"Created dummy audio file: {dummy_audio_path} ({duration}s)")
        except Exception as e:
            print(f"Could not create dummy audio file '{dummy_audio_path}': {e}. Test will likely fail.")
            dummy_audio_path = None 

    task_id = None
    if dummy_audio_path and os.path.exists(dummy_audio_path):
        print(f"\nAttempting to start transcription task for '{dummy_audio_path}'...")
        task_specific_generate_kwargs = {"hotword": "测试"} 
        start_response = transcriber.start_transcription_task(
            audio_path=dummy_audio_path,
            model_generate_kwargs=task_specific_generate_kwargs
        )
        print(f"Start task response: {start_response}")

        if start_response.get("status") == "processing_started":
            task_id = start_response["task_id"]
            print(f"Task {task_id} started. Polling for status (up to {20*2} seconds)...")

            max_polls = 200
            poll_interval = 5 
            for i in range(max_polls):
                time.sleep(poll_interval)
                status_response = transcriber.get_task_status(task_id)
                current_task_status = status_response.get('status')
                print(f"Poll {i+1}/{max_polls} - Task Status: {current_task_status}")
                
                if current_task_status == "completed":
                    print("Transcription completed!")
                    result_response = transcriber.get_transcription_result(task_id)
                    print("--- Transcription Result (Detailed) ---")
                    raw_result = result_response.get("result")
                    
                    if isinstance(raw_result, list): # Expected: list of segments
                        for idx, segment in enumerate(raw_result):
                            print(f"  Segment {idx+1}:")
                            if isinstance(segment, dict):
                                segment_text = segment.get('text', 'N/A')
                                print(f"    Text: \"{segment_text}\"")
                                if 'start' in segment and 'end' in segment:
                                    print(f"    Segment Time: [{segment['start']}ms - {segment['end']}ms]")
                                
                                # Check for word-level timestamps (common key: 'timestamp' or 'words')
                                word_timestamps = segment.get('timestamp') # Often list of [start, end] per word
                                if not word_timestamps and 'words' in segment: # Alternative structure
                                    word_timestamps = segment.get('words') # Often list of dicts {'text': word, 'start': s, 'end': e}

                                if word_timestamps:
                                    print(f"    Word Timestamps:")
                                    if all(isinstance(wt, list) and len(wt) == 2 for wt in word_timestamps): # [[start, end], ...]
                                        # This format usually doesn't include word text directly with timestamps.
                                        # The main 'text' would need to be split and aligned if words are needed.
                                        # For now, just print the time ranges.
                                        for w_idx, ts in enumerate(word_timestamps):
                                            print(f"      - Word {w_idx+1} Time: [{ts[0]}ms - {ts[1]}ms]")
                                    elif all(isinstance(wt, dict) and 'text' in wt and 'start' in wt and 'end' in wt for wt in word_timestamps): # [{'text': w, 'start': s, 'end': e}, ...]
                                        for wt_dict in word_timestamps:
                                            print(f"      - \"{wt_dict['text']}\" Time: [{wt_dict['start']}ms - {wt_dict['end']}ms]")
                                    else:
                                        print(f"      - (Timestamps format not recognized for detailed print: {word_timestamps})")
                                else:
                                    print("    (No detailed word timestamps found in this segment structure)")
                            else:
                                print(f"    (Segment is not a dictionary: {segment})")
                    else: # If not a list, print raw
                        print(f"  Raw Result (not a list of segments): {raw_result}")
                    print("-----------------------------------")
                    break
                elif current_task_status == "failed":
                    print("Transcription failed.")
                    result_response = transcriber.get_transcription_result(task_id)
                    print(f"Error details: {result_response.get('error_details')}")
                    break
                elif current_task_status == "error": 
                    print(f"Error fetching status: {status_response.get('message')}")
                    break
            else: 
                print("Max polls reached, task might still be processing or stuck.")

    non_existent_audio = "non_existent_audio_for_transcriber.wav"
    print(f"\nAttempting to start transcription for non-existent file: '{non_existent_audio}'...")
    start_response_invalid = transcriber.start_transcription_task(audio_path=non_existent_audio)
    print(f"Start task response (invalid file): {start_response_invalid}")

    fake_task_id = "fake-task-id"
    print(f"\nAttempting to get status for fake task ID: '{fake_task_id}'...")
    status_fake = transcriber.get_task_status(fake_task_id)
    print(f"Status for fake task: {status_fake}")

    print(f"\nAttempting to get result for fake task ID: '{fake_task_id}'...")
    result_fake = transcriber.get_transcription_result(fake_task_id)
    print(f"Result for fake task: {result_fake}")
    
    if task_id: 
        print(f"\nFinal status check for task {task_id}:")
        final_status = transcriber.get_task_status(task_id)
        print(final_status)
        if final_status.get("status") == "completed":
            print(f"\nFinal result (brief) for task {task_id}: First segment text: '{final_status.get('result', [{}])[0].get('text', 'N/A')}'")


    if dummy_audio_path and os.path.exists(dummy_audio_path):
        try:
            os.remove(dummy_audio_path)
            print(f"\nCleaned up {dummy_audio_path}")
        except Exception as e:
            print(f"Error cleaning up {dummy_audio_path}: {e}")
    
    print("\nSpeechTranscriber refined tests finished.")