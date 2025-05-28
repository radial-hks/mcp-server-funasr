import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import time
import uuid
import soundfile as sf
import numpy as np

# Ensure MCPServer is in path for imports if tests are run directly
# This might not be needed if running with `python -m unittest discover` from project root
import sys
# Assuming the tests are run from the directory containing MCPServer or MCPServer is in PYTHONPATH
# For robustness, especially if running this file directly:
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# However, the prompt implies the worker handles imports, so this might not be strictly necessary.

from MCPServer.speech_transcriber import SpeechTranscriber
from MCPServer.audio_processor import AudioProcessor


class TestSpeechTranscriber(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_transcriber_files"
        os.makedirs(self.test_dir, exist_ok=True)
        self.dummy_audio_path = os.path.join(self.test_dir, "dummy_audio.wav")
        samplerate = 16000
        duration = 0.2 # Increased duration from 0.01s
        frequency = 440
        t = np.linspace(0, duration, int(samplerate * duration), False, dtype=np.float32)
        data = 0.1 * np.sin(2 * np.pi * frequency * t)
        sf.write(self.dummy_audio_path, data, samplerate, format='WAV', subtype='PCM_16')

        # Default model for testing - can be overridden in specific tests
        # Using a real model name here as AutoModel might try to validate it
        # even if it's mocked later for the 'generate' part.
        self.test_model_name = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        # self.test_model_name = "dummy-asr-model" # Use if AutoModel validation is an issue

    def tearDown(self):
        if os.path.exists(self.dummy_audio_path):
            os.remove(self.dummy_audio_path)
        if os.path.exists(self.test_dir):
            if not os.listdir(self.test_dir):
                os.rmdir(self.test_dir)
            else:
                for f in os.listdir(self.test_dir):
                    os.remove(os.path.join(self.test_dir, f))
                os.rmdir(self.test_dir)

    @patch('funasr.AutoModel')
    def test_init_successful_model_load(self, mock_auto_model):
        mock_model_instance = MagicMock()
        mock_auto_model.return_value = mock_model_instance
        
        transcriber = SpeechTranscriber(default_model_name=self.test_model_name, ncpu=1)
        self.assertIsNotNone(transcriber.model)
        self.assertEqual(transcriber.current_model_name, self.test_model_name)
        mock_auto_model.assert_called_once_with(model=self.test_model_name, device="cpu", ncpu=1)

    @patch('funasr.AutoModel')
    def test_init_failed_model_load(self, mock_auto_model):
        mock_auto_model.side_effect = Exception("Failed to load model")
        # Suppress print during test
        with patch('builtins.print') as mock_print:
            transcriber = SpeechTranscriber(default_model_name="fail-model", ncpu=1, default_model_load_kwargs={})
            self.assertIsNone(transcriber.model)
            self.assertIsNone(transcriber.current_model_name) # current_model_name is set only on successful load
            
            # Check if any print call contained the expected message parts
            found_print = False
            for call_args in mock_print.call_args_list:
                arg_str = str(call_args)
                # The error message from load_model is "Error loading model '{model_name}': {e}"
                # The print in __init__ is f"FATAL: Initial model load failed for {self.default_model_name}. Error: {load_status['message']}"
                # So the combined message will have "Error: Error loading model..."
                if "FATAL: Initial model load failed for fail-model" in arg_str and "Error: Error loading model 'fail-model': Failed to load model" in arg_str:
                    found_print = True
                    break
            self.assertTrue(found_print, f"Expected FATAL error message was not printed. Actual calls: {mock_print.call_args_list}")

    @patch('funasr.AutoModel')
    def test_load_model_success(self, mock_auto_model):
        mock_initial_model = MagicMock()
        mock_new_model = MagicMock()
        mock_auto_model.side_effect = [mock_initial_model, mock_new_model]

        transcriber = SpeechTranscriber(default_model_name="initial-model", ncpu=1) # Initial load
        self.assertEqual(transcriber.current_model_name, "initial-model")
        
        result = transcriber.load_model("new-model", device="cpu", model_load_kwargs={"ncpu": 2})
        self.assertEqual(result["status"], "success")
        self.assertIn("Model 'new-model' loaded successfully", result["message"])
        self.assertEqual(transcriber.current_model_name, "new-model")
        self.assertEqual(transcriber.ncpu, 2)
        self.assertEqual(mock_auto_model.call_count, 2)
        mock_auto_model.assert_called_with(model="new-model", device="cpu", ncpu=2)


    @patch('funasr.AutoModel')
    def test_load_model_failure(self, mock_auto_model):
        mock_initial_model = MagicMock()
        mock_auto_model.side_effect = [mock_initial_model, Exception("Load failed")]
        
        transcriber = SpeechTranscriber(default_model_name="initial-model", ncpu=1)
        self.assertIsNotNone(transcriber.model) # Initial model should be loaded

        with patch('builtins.print') as mock_print:
            result = transcriber.load_model("fail-model")
            self.assertEqual(result["status"], "error")
            self.assertIn("Error loading model 'fail-model': Load failed", result["message"])
            # Model should remain the initially loaded one
            self.assertEqual(transcriber.current_model_name, "initial-model") 
            self.assertIsNotNone(transcriber.model) # Should still be the initial model
            mock_print.assert_any_call("Error loading model 'fail-model': Load failed")

    @patch.object(AudioProcessor, 'validate_audio')
    @patch('funasr.AutoModel') # Mock AutoModel for the transcriber's __init__
    def test_start_transcription_invalid_audio(self, mock_auto_model_init, mock_validate_audio):
        mock_auto_model_init.return_value = MagicMock() # Successful init
        mock_validate_audio.return_value = {"status": "invalid", "message": "Invalid audio", "details": None}
        
        transcriber = SpeechTranscriber(default_model_name=self.test_model_name, ncpu=1)
        result = transcriber.start_transcription_task("invalid_path.wav")
        
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["task_id"])
        self.assertEqual(result["message"], "Invalid audio")

    @patch('funasr.AutoModel')
    @patch.object(AudioProcessor, 'validate_audio') # Mock AudioProcessor within SpeechTranscriber
    def test_start_transcription_task_id_generation(self, mock_validate_audio, mock_auto_model):
        mock_model_instance = MagicMock()
        mock_auto_model.return_value = mock_model_instance
        mock_validate_audio.return_value = {"status": "valid", "details": {}}

        transcriber = SpeechTranscriber(default_model_name=self.test_model_name, ncpu=1)
        
        # Mock the _transcribe thread to prevent it from actually running FunASR
        with patch.object(transcriber, '_transcribe', MagicMock()) as mock_transcribe_method:
            result = transcriber.start_transcription_task(self.dummy_audio_path)
            self.assertEqual(result["status"], "processing_started")
            self.assertIsNotNone(result["task_id"])
            try:
                uuid.UUID(result["task_id"]) # Check if it's a valid UUID
            except ValueError:
                self.fail("Task ID is not a valid UUID")
            # Check if _transcribe was scheduled with the correct args
            mock_transcribe_method.assert_called_once() 


    @patch('funasr.AutoModel')
    @patch.object(AudioProcessor, 'validate_audio')
    def test_start_transcription_model_switch_logic(self, mock_validate_audio, mock_auto_model):
        mock_model_initial = MagicMock(name="InitialModel")
        mock_model_new = MagicMock(name="NewModel")
        # Configure AutoModel to return the initial model, then the new model on subsequent calls
        mock_model_initial = MagicMock(name="InitialModel")
        mock_model_new = MagicMock(name="NewModel")
        # Configure AutoModel to return the initial model, then the new model on subsequent calls
        mock_auto_model.side_effect = [mock_model_initial, mock_model_new, mock_model_initial] # load, switch, switch back

        mock_validate_audio.return_value = {"status": "valid", "details": {}}
        
        # Ensure default_model_load_kwargs in __init__ results in {'ncpu': 1}
        transcriber = SpeechTranscriber(default_model_name="initial-model", ncpu=1, default_model_load_kwargs={})
        self.assertEqual(transcriber.current_model_name, "initial-model")
        self.assertEqual(transcriber.model, mock_model_initial)
        # After init, current_model_load_kwargs should include ncpu if device is cpu
        self.assertEqual(transcriber.current_model_load_kwargs, {'ncpu': 1})


        # Mock the _transcribe thread
        with patch.object(transcriber, '_transcribe', MagicMock()):
            # Task with a new model name. This will call load_model internally.
            # load_model will use self.device ("cpu") and self.current_model_load_kwargs ({'ncpu': 1})
            # because model_load_kwargs is not passed to start_transcription_task.
            # The load_model method will then ensure 'ncpu' is in the effective_model_load_kwargs.
            transcriber.start_transcription_task(self.dummy_audio_path, model_name="new-model")
            self.assertEqual(mock_auto_model.call_count, 2) # Initial load + new model load
            # The internal call to load_model for "new-model" uses current_model_load_kwargs from "initial-model"
            # which should be {'ncpu': 1} after initialization.
            mock_auto_model.assert_called_with(model="new-model", device="cpu", ncpu=1) 
            self.assertEqual(transcriber.current_model_name, "new-model")
            self.assertEqual(transcriber.model, mock_model_new)

            # Task with no model name (should use current "new-model")
            transcriber.start_transcription_task(self.dummy_audio_path)
            self.assertEqual(mock_auto_model.call_count, 2) # No new model load
            self.assertEqual(transcriber.current_model_name, "new-model")

            # Task switching back to initial model
            transcriber.start_transcription_task(self.dummy_audio_path, model_name="initial-model")
            self.assertEqual(mock_auto_model.call_count, 3) # initial, new, initial_again
            mock_auto_model.assert_called_with(model="initial-model", device="cpu", ncpu=1)
            self.assertEqual(transcriber.current_model_name, "initial-model")


    @patch('funasr.AutoModel') # For __init__
    def test_get_status_non_existent_task(self, mock_auto_model_init):
        mock_auto_model_init.return_value = MagicMock()
        transcriber = SpeechTranscriber(default_model_name=self.test_model_name, ncpu=1)
        result = transcriber.get_task_status("fake-task-id")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Task ID not found.")

    @patch('funasr.AutoModel') # For __init__
    def test_get_result_non_existent_task(self, mock_auto_model_init):
        mock_auto_model_init.return_value = MagicMock()
        transcriber = SpeechTranscriber(default_model_name=self.test_model_name, ncpu=1)
        result = transcriber.get_transcription_result("fake-task-id")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Task ID not found.")

    # Conditional Integration Test: This test might be slow due to model loading.
    # It uses the actual FunASR model specified.
    # To make it runnable in CI, ensure the model is small or cached, or skip this test.
    # For now, it will use the default test_model_name which is Paraformer.
    # @unittest.skipIf(os.environ.get("CI") == "true", "Skipping slow integration test in CI")
    def test_integration_transcription_flow(self):
        # This test uses the actual SpeechTranscriber without mocking funasr.AutoModel for transcription part
        # It will download/load the model if not cached.
        # Using a very small ncpu count to be friendlier on shared resources for tests.
        # Ensure default_model_load_kwargs are sensible for the model.
        load_kwargs = {
            "vad_model": "fsmn-vad", # Using ModelScope ID for consistency
            "punc_model": "ct-punc"  # Using ModelScope ID
        }
        transcriber = SpeechTranscriber(
            default_model_name=self.test_model_name, # "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            ncpu=1, # Use minimal ncpu for testing
            default_model_load_kwargs=load_kwargs,
            default_model_generate_kwargs={"batch_size_s": 10} # Smaller batch for tiny audio
        )

        if not transcriber.model:
            self.skipTest(f"Default model {self.test_model_name} could not be loaded. Skipping integration test.")

        start_time = time.time()
        result = transcriber.start_transcription_task(self.dummy_audio_path)
        load_and_start_duration = time.time() - start_time
        print(f"Integration test: Model loading (if any) and task start took {load_and_start_duration:.2f}s")


        self.assertEqual(result["status"], "processing_started")
        task_id = result["task_id"]

        max_wait_time = 60  # seconds, Paraformer can take a while to load first time
        poll_interval = 1
        waited_time = 0
        final_status = None

        for _ in range(int(max_wait_time / poll_interval)):
            time.sleep(poll_interval)
            waited_time += poll_interval
            status_result = transcriber.get_task_status(task_id)
            final_status = status_result.get("status")
            if final_status == "completed":
                break
            if final_status == "failed":
                self.fail(f"Transcription task failed: {status_result.get('error_message')}")
        
        print(f"Integration test: Waited {waited_time}s for completion.")
        self.assertEqual(final_status, "completed", f"Task did not complete within {max_wait_time}s. Last status: {final_status}")

        transcription_result_data = transcriber.get_transcription_result(task_id)
        self.assertEqual(transcription_result_data["status"], "completed")
        self.assertIsNotNone(transcription_result_data["result"])
        # Result for a 0.01s sine wave will likely be empty or a very short, possibly odd transcription.
        # The key is that it completed.
        print(f"Integration test: Transcription result: {transcription_result_data['result']}")


if __name__ == "__main__":
    unittest.main()
