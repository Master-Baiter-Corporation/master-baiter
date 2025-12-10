import os
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np

def split_audio_into_chunks(input_folder, output_folder=None, chunk_length_sec=1.0, sample_rate=16000):
    """
    Split audio files into 1-second chunks using librosa.
    
    Args:
        input_folder: Path to folder containing WAV files to split
        output_folder: Path to save chunks (if None, saves in same folder)
        chunk_length_sec: Length of each chunk in seconds (1.0 = 1 second)
        sample_rate: Sample rate for output files
    """
    input_path = Path(input_folder)
    
    # Use same folder if output not specified
    if output_folder is None:
        output_folder = input_folder
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all WAV files
    wav_files = list(input_path.glob('*.wav'))
    
    if not wav_files:
        print(f"No WAV files found in {input_folder}")
        return
    
    print(f"Found {len(wav_files)} WAV file(s) to process")
    
    total_chunks = 0
    
    # Process each WAV file
    for file_idx, wav_file in enumerate(wav_files, 1):
        print(f"\nProcessing: {wav_file.name}")
        
        # Load audio file
        audio, sr = librosa.load(wav_file, sr=sample_rate, mono=True)
        duration_sec = len(audio) / sr
        
        print(f"Duration: {duration_sec:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        
        # Calculate number of samples per chunk
        samples_per_chunk = int(chunk_length_sec * sr)
        
        # Calculate number of chunks
        num_chunks = len(audio) // samples_per_chunk
        
        print(f"Creating {num_chunks} chunks...")
        
        # Split into chunks
        for chunk_idx in range(int(num_chunks)):
            start_sample = chunk_idx * samples_per_chunk
            end_sample = start_sample + samples_per_chunk
            
            # Extract chunk
            chunk = audio[start_sample:end_sample]
            
            # Create filename: original_name_chunk_001.wav
            original_name = wav_file.stem
            chunk_filename = f"{original_name}_chunk_{chunk_idx + 1:03d}.wav"
            chunk_path = output_path / chunk_filename
            
            # Save chunk as WAV
            sf.write(chunk_path, chunk, sr)
            total_chunks += 1
            
            if (chunk_idx + 1) % 10 == 0:
                print(f"  Created {chunk_idx + 1} chunks...")
        
        print(f"  Completed: {int(num_chunks)} chunks from {wav_file.name}")
    
    print(f"\n{'='*50}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Saved to: {output_path}")
    print(f"{'='*50}")


# ==================== USAGE ====================

if __name__ == "__main__":
    # Configuration
    SILENCE_FOLDER = "audios_cleans/silence"  # Change this path
    
    # Option 1: Save chunks in the same folder
    split_audio_into_chunks(SILENCE_FOLDER)
    
    # Option 2: Save chunks to a different folder (keeps originals)
    # split_audio_into_chunks(SILENCE_FOLDER, output_folder="path/to/output/silence")
