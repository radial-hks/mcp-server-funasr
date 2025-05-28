import unittest
import os
import soundfile as sf
import numpy as np
from MCPServer.audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor()
        self.test_dir = "test_audio_files"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a dummy valid WAV file
        self.valid_audio_path = os.path.join(self.test_dir, "valid_audio.wav")
        samplerate = 16000
        duration = 0.1 # Short duration
        frequency = 440
        t = np.linspace(0, duration, int(samplerate * duration), False, dtype=np.float32)
        data = 0.5 * np.sin(2 * np.pi * frequency * t)
        sf.write(self.valid_audio_path, data, samplerate, format='WAV', subtype='PCM_16')

        # Create a dummy non-audio file
        self.non_audio_path = os.path.join(self.test_dir, "not_audio.txt")
        with open(self.non_audio_path, "w") as f:
            f.write("This is a test text file.")

        # Path for a non-existent file
        self.non_existent_path = os.path.join(self.test_dir, "non_existent.wav")

    def tearDown(self):
        if os.path.exists(self.valid_audio_path):
            os.remove(self.valid_audio_path)
        if os.path.exists(self.non_audio_path):
            os.remove(self.non_audio_path)
        if os.path.exists(self.test_dir):
            # Check if directory is empty before removing (optional, good practice)
            if not os.listdir(self.test_dir):
                 os.rmdir(self.test_dir)
            else: # If other files were created by failed tests, remove them too
                for f in os.listdir(self.test_dir):
                    os.remove(os.path.join(self.test_dir, f))
                os.rmdir(self.test_dir)


    def test_validate_audio_valid_file(self):
        result = self.processor.validate_audio(self.valid_audio_path)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["message"], "Audio file is valid.")
        self.assertIsNotNone(result["details"])
        self.assertEqual(result["details"]["samplerate"], 16000)
        self.assertEqual(result["details"]["channels"], 1)
        self.assertAlmostEqual(result["details"]["duration"], 0.1, places=2)

    def test_validate_audio_non_existent_file(self):
        result = self.processor.validate_audio(self.non_existent_path)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("File not found", result["message"])
        self.assertIsNone(result["details"])

    def test_validate_audio_non_audio_file(self):
        result = self.processor.validate_audio(self.non_audio_path)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("not a valid audio file or is corrupted", result["message"])
        self.assertIsNone(result["details"])

    def test_validate_audio_unreadable_file(self):
        # Create a file and then make it unreadable (if permissions allow)
        unreadable_path = os.path.join(self.test_dir, "unreadable_audio.wav")
        sf.write(unreadable_path, [0.0], 16000)
        
        original_mode = os.stat(unreadable_path).st_mode
        try:
            os.chmod(unreadable_path, 0o000) # Remove all permissions
            result = self.processor.validate_audio(unreadable_path)
            # Behavior might vary: on some OS/filesystems, this might still be readable by owner/root
            # or sf.info might fail differently. The os.access check is key.
            if not os.access(unreadable_path, os.R_OK): # If truly unreadable by current user
                self.assertEqual(result["status"], "invalid")
                self.assertIn("is not readable", result["message"])
            else:
                # If it was readable despite chmod (e.g. root user), it might pass os.access
                # but sf.info might still fail if there's an issue with the file handle.
                # For this test, we primarily rely on the os.access check.
                print(f"Warning: File {unreadable_path} was still readable despite chmod 000. Test for unreadability might not be effective.")
                self.assertIn(result["status"], ["invalid", "valid"]) # Allow for systems where it's hard to make truly unreadable
        finally:
            os.chmod(unreadable_path, original_mode) # Restore permissions
            if os.path.exists(unreadable_path):
                os.remove(unreadable_path)


if __name__ == "__main__":
    unittest.main()
