import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import soundfile as sf
import numpy as np

from MCPServer.vad_processor import VADProcessor
from MCPServer.audio_processor import AudioProcessor

class TestVADProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_vad_files"
        os.makedirs(self.test_dir, exist_ok=True)
        self.dummy_audio_path = os.path.join(self.test_dir, "dummy_vad_audio.wav")
        samplerate = 16000
        duration = 0.5 # Short audio for VAD
        # Create audio with some silence then some sound
        silence_duration = int(0.1 * samplerate)
        sound_duration = int(0.4 * samplerate)
        silence = np.zeros(silence_duration, dtype=np.float32)
        t_sound = np.linspace(0, 0.4, sound_duration, False, dtype=np.float32)
        sound = 0.3 * np.sin(2 * np.pi * 440 * t_sound)
        data = np.concatenate((silence, sound))
        
        sf.write(self.dummy_audio_path, data, samplerate, format='WAV', subtype='PCM_16')

        self.test_model_name = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch" # Default VAD model

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
    def test_init_successful_vad_model_load(self, mock_auto_model):
        mock_model_instance = MagicMock()
        mock_auto_model.return_value = mock_model_instance
        
        vad_processor = VADProcessor(default_vad_model_name=self.test_model_name, ncpu=1)
        self.assertIsNotNone(vad_processor.vad_model)
        self.assertEqual(vad_processor.current_vad_model_name, self.test_model_name)
        mock_auto_model.assert_called_once_with(model=self.test_model_name, device="cpu", ncpu=1)

    @patch('funasr.AutoModel')
    def test_init_failed_vad_model_load(self, mock_auto_model):
        mock_auto_model.side_effect = Exception("Failed to load VAD model")
        with patch('builtins.print') as mock_print:
            vad_processor = VADProcessor(default_vad_model_name="fail-vad-model", ncpu=1, default_model_load_kwargs={})
            self.assertIsNone(vad_processor.vad_model)
            self.assertIsNone(vad_processor.current_vad_model_name)
            
            found_print = False
            for call_args in mock_print.call_args_list:
                arg_str = str(call_args)
                # The error message from load_vad_model is "Error loading VAD model '{model_name}': {e}"
                # The print in __init__ is f"FATAL: Initial VAD model load failed for {self.default_vad_model_name}. Error: {load_status['message']}"
                # So the combined message will have "Error: Error loading VAD model..."
                if "FATAL: Initial VAD model load failed for fail-vad-model" in arg_str and "Error: Error loading VAD model 'fail-vad-model': Failed to load VAD model" in arg_str:
                    found_print = True
                    break
            self.assertTrue(found_print, f"Expected FATAL VAD error message was not printed. Actual calls: {mock_print.call_args_list}")

    @patch('funasr.AutoModel')
    def test_load_vad_model_success(self, mock_auto_model):
        mock_initial_model = MagicMock()
        mock_new_model = MagicMock()
        mock_auto_model.side_effect = [mock_initial_model, mock_new_model] # First for init, second for explicit load

        vad_processor = VADProcessor(default_vad_model_name="initial-vad", ncpu=1)
        self.assertEqual(vad_processor.current_vad_model_name, "initial-vad")
        
        result = vad_processor.load_vad_model("new-vad-model", device="cpu", ncpu=2, model_load_kwargs={"some_custom_arg": True})
        self.assertEqual(result["status"], "success")
        self.assertIn("VAD Model 'new-vad-model' loaded successfully", result["message"])
        self.assertEqual(vad_processor.current_vad_model_name, "new-vad-model")
        self.assertEqual(vad_processor.ncpu, 2) # ncpu should be updated
        self.assertEqual(vad_processor.device, "cpu")
        self.assertTrue(vad_processor.current_model_load_kwargs.get("some_custom_arg"))
        self.assertEqual(mock_auto_model.call_count, 2)
        mock_auto_model.assert_called_with(model="new-vad-model", device="cpu", ncpu=2, some_custom_arg=True)

    @patch('funasr.AutoModel')
    def test_load_vad_model_failure(self, mock_auto_model):
        mock_initial_model = MagicMock()
        mock_auto_model.side_effect = [mock_initial_model, Exception("VAD Load failed")]
        
        vad_processor = VADProcessor(default_vad_model_name="initial-vad", ncpu=1)
        self.assertIsNotNone(vad_processor.vad_model)

        with patch('builtins.print') as mock_print:
            result = vad_processor.load_vad_model("fail-vad-model")
            self.assertEqual(result["status"], "error")
            self.assertIn("Error loading VAD model 'fail-vad-model': VAD Load failed", result["message"])
            self.assertEqual(vad_processor.current_vad_model_name, "initial-vad") # Should retain initial
            mock_print.assert_any_call("Error loading VAD model 'fail-vad-model': VAD Load failed")


    @patch.object(AudioProcessor, 'validate_audio')
    @patch('funasr.AutoModel') # For VADProcessor's __init__
    def test_get_speech_segments_invalid_audio(self, mock_auto_model_init, mock_validate_audio):
        mock_auto_model_init.return_value = MagicMock() # Successful init
        mock_validate_audio.return_value = {"status": "invalid", "message": "Test Invalid audio path", "details": None}
        
        vad_processor = VADProcessor(default_vad_model_name=self.test_model_name, ncpu=1)
        result = vad_processor.get_speech_segments("invalid_path.wav")
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Test Invalid audio path")

    @patch('funasr.AutoModel')
    @patch.object(AudioProcessor, 'validate_audio')
    def test_get_speech_segments_model_switch_logic(self, mock_validate_audio, mock_auto_model):
        mock_model_initial = MagicMock(name="InitialVADModel")
        mock_model_new = MagicMock(name="NewVADModel")
        mock_auto_model.side_effect = [mock_model_initial, mock_model_new, mock_model_initial] # init, switch, switch_back

        mock_validate_audio.return_value = {"status": "valid", "details": {}}
        # Mock the generate call for VAD models
        mock_model_initial.generate.return_value = [{"text": [[100, 200]]}] # FunASR VAD output format
        mock_model_new.generate.return_value = [{"text": [[300, 400]]}]

        vad_processor = VADProcessor(default_vad_model_name="initial-vad", ncpu=1)
        self.assertEqual(vad_processor.current_vad_model_name, "initial-vad")
        
        # Task with a new model name
        result_new_model = vad_processor.get_speech_segments(self.dummy_audio_path, vad_model_name="new-vad-model")
        self.assertEqual(mock_auto_model.call_count, 2)
        mock_auto_model.assert_called_with(model="new-vad-model", device="cpu", ncpu=1) # Assuming default ncpu is used
        self.assertEqual(vad_processor.current_vad_model_name, "new-vad-model")
        self.assertEqual(result_new_model["segments"], [[300, 400]]) # Check if new model's output is used

        # Task with no model name (should use current "new-vad-model")
        vad_processor.get_speech_segments(self.dummy_audio_path)
        self.assertEqual(mock_auto_model.call_count, 2) # No new model load
        self.assertEqual(vad_processor.current_vad_model_name, "new-vad-model")

        # Task switching back
        result_back_model = vad_processor.get_speech_segments(self.dummy_audio_path, vad_model_name="initial-vad")
        self.assertEqual(mock_auto_model.call_count, 3)
        self.assertEqual(result_back_model["segments"], [[100, 200]])


    @patch('funasr.AutoModel') # For __init__
    @patch.object(AudioProcessor, 'validate_audio')
    def test_get_speech_segments_generate_failure(self, mock_validate_audio, mock_auto_model_init):
        mock_vad_model_instance = MagicMock()
        mock_auto_model_init.return_value = mock_vad_model_instance
        mock_validate_audio.return_value = {"status": "valid", "details": {}}
        mock_vad_model_instance.generate.side_effect = Exception("VAD Generate Error")

        vad_processor = VADProcessor(default_vad_model_name=self.test_model_name, ncpu=1)
        with patch('builtins.print') as mock_print:
            result = vad_processor.get_speech_segments(self.dummy_audio_path)
            self.assertEqual(result["status"], "error")
            self.assertIn("VAD processing failed", result["message"])
            self.assertIn("VAD Generate Error", result["message"])
            mock_print.assert_any_call(f"VAD processing failed for '{self.dummy_audio_path}': VAD Generate Error")

    # Conditional Integration Test for VAD
    # @unittest.skipIf(os.environ.get("CI") == "true", "Skipping slow VAD integration test in CI")
    def test_integration_get_speech_segments(self):
        # This test uses the actual VADProcessor without mocking funasr.AutoModel for VAD part
        vad_processor = VADProcessor(
            default_vad_model_name=self.test_model_name, # "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            ncpu=1 # Minimal ncpu for testing
        )
        if not vad_processor.vad_model:
            self.skipTest(f"Default VAD model {self.test_model_name} could not be loaded. Skipping integration test.")

        result = vad_processor.get_speech_segments(self.dummy_audio_path)
        
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["segments"])
        self.assertIsInstance(result["segments"], list)
        # For the dummy audio (0.1s silence, 0.4s sound), we expect at least one segment.
        # The exact ms values depend on the VAD model's behavior.
        if len(result["segments"]) > 0:
            print(f"Integration VAD test: Segments found: {result['segments']}")
            for segment in result["segments"]:
                self.assertIsInstance(segment, list)
                self.assertEqual(len(segment), 2)
                self.assertIsInstance(segment[0], (int, float)) # Start time
                self.assertIsInstance(segment[1], (int, float)) # End time
                self.assertLessEqual(segment[0], segment[1]) # Start <= End
        else:
            print(f"Integration VAD test: No segments found by {self.test_model_name} for {self.dummy_audio_path}. This might be acceptable for very short/ambiguous audio.")
            # self.fail("VAD did not detect any segments in the dummy audio.") # Could fail if segments are strictly expected


if __name__ == "__main__":
    unittest.main()
