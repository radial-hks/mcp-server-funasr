import os
import soundfile as sf

class AudioProcessor:
    def __init__(self):
        pass

    def validate_audio(self, file_path: str) -> dict:
        """
        Validates an audio file.

        Checks if the file exists, is readable, and is a valid audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            A dictionary with validation status, message, and details.
        """
        if not os.path.exists(file_path):
            return {
                "status": "invalid",
                "message": f"Error: File not found at '{file_path}'.",
                "details": None,
            }

        if not os.access(file_path, os.R_OK):
            return {
                "status": "invalid",
                "message": f"Error: File at '{file_path}' is not readable.",
                "details": None,
            }

        try:
            # Check if it's a valid audio file and get its properties
            audio_info = sf.info(file_path)
            duration_seconds = audio_info.duration
            # More detailed duration formatting (optional)
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            milliseconds = int((duration_seconds - (minutes * 60 + seconds)) * 1000)
            formatted_duration = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

            return {
                "status": "valid",
                "message": "Audio file is valid.",
                "details": {
                    "samplerate": audio_info.samplerate,
                    "channels": audio_info.channels,
                    "duration": duration_seconds,
                    "formatted_duration": formatted_duration,
                    "format": audio_info.format,
                    "subtype": audio_info.subtype,
                },
            }
        except sf.LibsndfileError as e:
            return {
                "status": "invalid",
                "message": f"Error: File at '{file_path}' is not a valid audio file or is corrupted. Details: {e}",
                "details": None,
            }
        except Exception as e:
            return {
                "status": "invalid",
                "message": f"An unexpected error occurred while validating the audio file at '{file_path}'. Details: {e}",
                "details": None,
            }

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    processor = AudioProcessor()
    audio_file_path = "E:\Code\PythonDir\MCP\mcp-server-funasr\Data\_20240821153822.mp3"
    # Test with a valid file (assuming you have a 'test.wav' in the same directory or provide a full path)
    # Create a dummy wav file for testing if one doesn't exist
    if not os.path.exists(audio_file_path):
        try:
            # Create a dummy mono WAV file, 1 second duration, 16kHz sample rate
            samplerate = 16000
            duration = 1
            frequency = 440 # A4 note
            import numpy as np
            t = np.linspace(0, duration, int(samplerate * duration), False)
            data = 0.5 * np.sin(2 * np.pi * frequency * t)
            sf.write(audio_file_path, data, samplerate)
            print("Created dummy 'test111.wav' for testing.")
        except Exception as e:
            print(f"Could not create dummy 'test.wav': {e}")


    valid_result = processor.validate_audio(audio_file_path)
    print(f"Validation for {audio_file_path}: {valid_result}")

    # Test with a non-existent file
    non_existent_result = processor.validate_audio("non_existent_audio.wav")
    print(f"Validation for 'non_existent_audio.wav': {non_existent_result}")

    # Test with a non-audio file (e.g., this Python script itself)
    # Create a dummy text file
    if not os.path.exists("not_an_audio_file.txt"):
        with open("not_an_audio_file.txt", "w") as f:
            f.write("This is not an audio file.")
        print("Created dummy 'not_an_audio_file.txt' for testing.")

    non_audio_result = processor.validate_audio("not_an_audio_file.txt")
    print(f"Validation for 'not_an_audio_file.txt': {non_audio_result}")

    # Test with a file that might not be readable (permissions issue - harder to test reliably in sandbox)
    # You might need to manually set permissions for such a test case.
    # For now, we'll assume this case is covered by the os.access check.
    # Example: Create a file and then remove read permissions
    # if not os.path.exists("unreadable.wav"):
    #     sf.write("unreadable.wav", [0.0], 16000) # Minimal valid wav
    #     os.chmod("unreadable.wav", 0o222) # Write-only permissions for owner
    #     print("Created 'unreadable.wav' with restricted permissions.")
    # unreadable_result = processor.validate_audio("unreadable.wav")
    # print(f"Validation for 'unreadable.wav': {unreadable_result}")
    # if os.path.exists("unreadable.wav"):
    #     os.chmod("unreadable.wav", 0o644) # Restore permissions
    #     os.remove("unreadable.wav")

    # Clean up dummy files
    if os.path.exists("test.wav"):
        os.remove("test.wav")
    if os.path.exists("not_an_audio_file.txt"):
        os.remove("not_an_audio_file.txt")