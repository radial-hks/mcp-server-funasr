import funasr
import os
from .audio_processor import AudioProcessor # Assuming AudioProcessor is in the same directory

# For example usage
import soundfile as sf
import numpy as np
import time

if "MODELSCOPE_API_TOKEN" not in os.environ:
    print("Note: MODELSCOPE_API_TOKEN environment variable not set. Some VAD models might require it for download.")

class VADProcessor:
    def __init__(self, default_vad_model_name: str = "fsmn-vad", 
                 device: str = "cpu", 
                 ncpu: int = 4, 
                 default_model_load_kwargs: dict = None,
                 default_model_generate_kwargs: dict = None):
        """
        Constructor for VADProcessor.

        Args:
            default_vad_model_name (str): Default FunASR VAD model to load.
            device (str): Device to run the model on (e.g., "cpu", "cuda:0").
            ncpu (int): Number of CPU threads for CPU inference.
            default_model_load_kwargs (dict, optional): Default kwargs for VAD model loading (funasr.AutoModel).
                                                       'ncpu' will be added/overridden if device is 'cpu'.
            default_model_generate_kwargs (dict, optional): Default kwargs for vad_model.generate().
        """
        self.default_vad_model_name = default_vad_model_name
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

        self.current_vad_model_name = None
        self.vad_model = None
        self.current_model_load_kwargs = None # Kwargs used for loading current model
        self.audio_processor = AudioProcessor()

        # Initial VAD model load
        load_status = self.load_vad_model(
            model_name=self.default_vad_model_name, 
            device=self.device, 
            model_load_kwargs=self.default_model_load_kwargs
        )
        if load_status["status"] == "error":
            # This error should ideally be logged or handled more gracefully.
            print(f"FATAL: Initial VAD model load failed for {self.default_vad_model_name}. Error: {load_status['message']}")
            # self.vad_model will remain None if loading fails.

    def load_vad_model(self, model_name: str, device: str = None, ncpu: int = None, model_load_kwargs: dict = None) -> dict:
        """
        Loads or reloads a VAD model.

        Args:
            model_name (str): The FunASR VAD model to load.
            device (str, optional): Device for the model. Defaults to instance's current device.
            ncpu (int, optional): Number of CPU threads. Defaults to instance's current ncpu. Used if device is 'cpu'.
            model_load_kwargs (dict, optional): Kwargs for funasr.AutoModel.

        Returns:
            dict: Status of the model loading operation.
        """
        effective_device = device if device is not None else self.device
        effective_ncpu = ncpu if ncpu is not None else self.ncpu # ncpu from arg takes precedence
        
        # Prioritize model_load_kwargs from args, then instance default, then empty dict
        final_model_load_kwargs = self.default_model_load_kwargs.copy()
        if model_load_kwargs is not None:
            final_model_load_kwargs.update(model_load_kwargs)

        if effective_device == "cpu":
            # Ensure ncpu is correctly set in kwargs for CPU devices
            # If ncpu is explicitly in final_model_load_kwargs, it takes precedence.
            # Otherwise, use effective_ncpu (derived from arg or instance self.ncpu).
            if 'ncpu' not in final_model_load_kwargs or final_model_load_kwargs['ncpu'] != effective_ncpu :
                 final_model_load_kwargs['ncpu'] = effective_ncpu
        else: # For non-CPU devices, remove ncpu if it's there
            if 'ncpu' in final_model_load_kwargs:
                del final_model_load_kwargs['ncpu']
        
        try:
            self.vad_model = funasr.AutoModel(model=model_name, device=effective_device, **final_model_load_kwargs)
            self.current_vad_model_name = model_name
            self.device = effective_device # Update instance device
            self.current_model_load_kwargs = final_model_load_kwargs # Store the actual kwargs used
            if effective_device == "cpu" and 'ncpu' in final_model_load_kwargs:
                 self.ncpu = final_model_load_kwargs['ncpu'] # Update instance ncpu if specified for CPU

            return {"status": "success", "message": f"VAD Model '{model_name}' loaded successfully on {self.device} with load_kwargs: {final_model_load_kwargs}."}
        except Exception as e:
            error_message = f"Error loading VAD model '{model_name}': {e}"
            print(error_message)
            # Potentially revert to a known good state or None
            # If a model was already loaded, it remains. If not, self.vad_model is None.
            return {"status": "error", "message": error_message}

    def get_speech_segments(self, audio_path: str, vad_model_name: str = None, 
                            model_load_kwargs: dict = None, 
                            model_generate_kwargs: dict = None) -> dict:
        """
        Identifies speech segments in an audio file using VAD.

        Args:
            audio_path (str): Path to the audio file.
            vad_model_name (str, optional): Specific VAD model to use. Defaults to the current one.
            model_load_kwargs (dict, optional): Kwargs for loading the VAD model if different from current.
            model_generate_kwargs (dict, optional): Kwargs for VAD model.generate().

        Returns:
            dict: Contains status, segments, and other info.
        """
        validation_result = self.audio_processor.validate_audio(audio_path)
        if validation_result["status"] == "invalid":
            return {"status": "error", "message": validation_result["message"], "details": validation_result.get("details")}

        # Model selection and loading
        if vad_model_name and vad_model_name != self.current_vad_model_name:
            print(f"Switching VAD model from '{self.current_vad_model_name}' to '{vad_model_name}'.")
            # Use provided model_load_kwargs for this specific load, or instance defaults if None
            load_kwargs_for_switch = model_load_kwargs if model_load_kwargs is not None else self.default_model_load_kwargs
            load_status = self.load_vad_model(model_name=vad_model_name, model_load_kwargs=load_kwargs_for_switch) # uses current instance device/ncpu unless overridden in load_kwargs_for_switch
            if load_status["status"] == "error":
                return {"status": "error", "message": f"Failed to switch to VAD model '{vad_model_name}'. Error: {load_status['message']}"}
        
        if not self.vad_model:
            return {"status": "error", "message": "No VAD model is loaded. Cannot process speech segments."}

        # Determine generate_kwargs: task-specific > instance default
        final_model_generate_kwargs = self.default_model_generate_kwargs.copy()
        if model_generate_kwargs is not None:
            final_model_generate_kwargs.update(model_generate_kwargs)
            
        try:
            # Perform VAD
            vad_result = self.vad_model.generate(input=audio_path, **final_model_generate_kwargs)
            # FunASR VAD typically returns list of [start_ms, end_ms] or dict with "text" key (which is segments)
            # We should ensure 'vad_result' is the actual list of segments.
            # Some VAD models in FunASR might return a list containing a dictionary like [{'text': [[0, 1000], [1500, 2500]]}]
            # or directly the list of segments. We should check the type.
            segments = [] # Default to empty list
            if isinstance(vad_result, list) and len(vad_result) > 0:
                first_item = vad_result[0]
                if isinstance(first_item, dict):
                    if "text" in first_item and isinstance(first_item["text"], list):
                        # Common format for some models, e.g., if it were ASR segments
                        segments = first_item["text"]
                    elif "value" in first_item and isinstance(first_item["value"], list):
                        # Format like [{'key': '...', 'value': [[start, end], ...]}]
                        segments = first_item["value"]
                    else: # Unrecognized dict structure in list
                        print(f"Warning: VAD output has a list of dicts, but keys 'text' or 'value' with list content not found. Raw first item: {first_item}")
                elif all(isinstance(item, list) and len(item) == 2 for item in vad_result):
                    # Direct list of segments [[start, end], ...]
                    segments = vad_result
                else: # List of something else
                    print(f"Warning: VAD output is a list, but not of segments or expected dicts. Raw output: {vad_result}")
            elif vad_result: # Not a list or empty list, but truthy (e.g. a single dict not in a list)
                 print(f"Warning: VAD output was not a list of segments or was empty. Raw output: {vad_result}")

            # Ensure all segment items are lists of two numbers if segments were found
            if not all(isinstance(s, list) and len(s) == 2 and isinstance(s[0], (int, float)) and isinstance(s[1], (int, float)) for s in segments):
                print(f"Warning: Parsed segments do not conform to [[start, end], ...]. Segments: {segments}. Raw VAD output: {vad_result}")
                # Depending on desired strictness, could set segments to [] or raise error
                # For now, pass through potentially malformed segments if parsing failed to correct.

            return {
                "status": "success",
                "segments": segments,
                "audio_path": audio_path,
                "vad_model_used": self.current_vad_model_name,
                "generate_kwargs_used": final_model_generate_kwargs,
                "audio_details": validation_result.get("details")
            }
        except Exception as e:
            error_message = f"VAD processing failed for '{audio_path}': {e}"
            print(error_message)
            return {"status": "error", "message": error_message, "audio_path": audio_path, "vad_model_used": self.current_vad_model_name}


if __name__ == '__main__':
    print("Attempting to initialize VADProcessor with default model...")
    # Default VAD model: "fsmn-vad" (ModelScope ID: "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    # This model usually has specific generate_kwargs it can accept, e.g., related to segment length.
    # For "fsmn-vad", `max_single_segment_time` is a load kwarg, not generate.
    # Generate kwargs for VAD are less common than for ASR, often defaults are fine.
    # Let's assume default_model_load_kwargs for VAD could be something like:
    # default_load_kwargs_for_vad = {"max_single_segment_time": 30000} # 30 seconds
    default_load_kwargs_for_vad = {} # Keep it simple for now
    default_generate_kwargs_for_vad = {} # VAD models usually don't need many generate_kwargs

    vad_processor = VADProcessor(
        default_vad_model_name="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        device="cpu",
        ncpu=2,
        default_model_load_kwargs=default_load_kwargs_for_vad,
        default_model_generate_kwargs=default_generate_kwargs_for_vad
    )

    if not vad_processor.vad_model:
        print("VADProcessor initialization failed to load the default model. Exiting.")
        exit()
    print(f"VADProcessor initialized. Current VAD model: {vad_processor.current_vad_model_name}")

    # Create a dummy audio file with some silence and speech
    dummy_vad_audio_path = "dummy_vad_test_audio.wav"
    samplerate = 16000
    if not os.path.exists(dummy_vad_audio_path):
        try:
            # 0.5s silence, 1s speech, 0.5s silence, 1s speech, 0.5s silence
            silence_duration1 = int(0.5 * samplerate)
            speech_duration1 = int(1.0 * samplerate)
            silence_duration2 = int(0.5 * samplerate)
            speech_duration2 = int(1.0 * samplerate)
            silence_duration3 = int(0.5 * samplerate)

            silence1 = np.zeros(silence_duration1, dtype=np.float32)
            
            t1 = np.linspace(0, 1.0, speech_duration1, False, dtype=np.float32)
            speech1 = 0.3 * np.sin(2 * np.pi * 330 * t1) # Speech-like segment 1
            
            silence2 = np.zeros(silence_duration2, dtype=np.float32)

            t2 = np.linspace(0, 1.0, speech_duration2, False, dtype=np.float32)
            speech2 = 0.4 * np.sin(2 * np.pi * 440 * t2) # Speech-like segment 2
            
            silence3 = np.zeros(silence_duration3, dtype=np.float32)

            data = np.concatenate((silence1, speech1, silence2, speech2, silence3))
            sf.write(dummy_vad_audio_path, data, samplerate, format='WAV', subtype='PCM_16')
            print(f"Created dummy VAD audio file: {dummy_vad_audio_path} (Total duration: {len(data)/samplerate:.2f}s)")
        except Exception as e:
            print(f"Could not create dummy VAD audio file '{dummy_vad_audio_path}': {e}. Test will likely fail.")
            dummy_vad_audio_path = None

    if dummy_vad_audio_path and os.path.exists(dummy_vad_audio_path):
        print(f"\nAttempting to get speech segments for '{dummy_vad_audio_path}'...")
        # Example: VAD models might accept 'offline_chunk_size' or other specific generate_kwargs
        # For fsmn-vad, it seems generate_kwargs are not typically needed for basic operation.
        segments_response = vad_processor.get_speech_segments(audio_path=dummy_vad_audio_path)
        print(f"Segments Response: {segments_response}")

        if segments_response["status"] == "success":
            print("Speech Segments (ms):")
            for segment in segments_response["segments"]:
                print(f"  Start: {segment[0]}, End: {segment[1]}")
        else:
            print(f"Error getting segments: {segments_response.get('message')}")

        # Test with specific generate_kwargs (if any are known for the VAD model)
        # For fsmn-vad, one documented generate kwarg is `cache` for streaming, not relevant here.
        # Another is `chunk_size` for streaming VAD.
        # Let's try an example if we knew one, e.g., if it took 'detection_threshold'
        # print("\nAttempting with example generate_kwargs...")
        # example_gen_kwargs = {"detection_threshold": 0.6} # This is a hypothetical kwarg
        # segments_response_custom = vad_processor.get_speech_segments(
        #     audio_path=dummy_vad_audio_path,
        #     model_generate_kwargs=example_gen_kwargs
        # )
        # print(f"Segments Response (custom_kwargs): {segments_response_custom}")


    # Test with a non-existent file
    non_existent_audio = "non_existent_vad_audio.wav"
    print(f"\nAttempting VAD for non-existent file: '{non_existent_audio}'...")
    response_invalid = vad_processor.get_speech_segments(audio_path=non_existent_audio)
    print(f"Response (invalid file): {response_invalid}")

    # Test dynamic model loading (if another VAD model is known and small)
    # FunASR Model Zoo lists "speech_seaco_scm_ns_aux_16k_pytorch" as another VAD related model,
    # but it might be for noise suppression or something else.
    # For simplicity, we'll skip dynamic loading test unless a clearly suitable, small alternative VAD model is identified.
    # print("\nAttempting to load a different VAD model (example)...")
    # load_resp = vad_processor.load_vad_model("another-vad-model-name") # Replace with actual if available
    # print(f"Load response for new VAD model: {load_resp}")
    # if load_resp["status"] == "success" and dummy_vad_audio_path and os.path.exists(dummy_vad_audio_path):
    #     segments_response_new_model = vad_processor.get_speech_segments(audio_path=dummy_vad_audio_path)
    #     print(f"Segments Response (new VAD model): {segments_response_new_model}")

    # Clean up dummy audio file
    if dummy_vad_audio_path and os.path.exists(dummy_vad_audio_path):
        try:
            os.remove(dummy_vad_audio_path)
            print(f"\nCleaned up {dummy_vad_audio_path}")
        except Exception as e:
            print(f"Error cleaning up {dummy_vad_audio_path}: {e}")
    
    print("\nVADProcessor tests finished.")
