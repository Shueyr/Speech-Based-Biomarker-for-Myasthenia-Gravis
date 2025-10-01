"""
Audio Processor Module
Handles audio preprocessing, VAD trimming, and segmentation into 10-second chunks.
"""

import torch
import torchaudio
import os
import sys


class AudioProcessor:
    """Processes audio files for MG classification."""
    
    def __init__(self, sample_rate=16000, segment_duration=10):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate (default: 16000 Hz)
            segment_duration: Duration of each segment in seconds (default: 10)
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load Silero VAD model
        torch.set_num_threads(1)
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad'
        )
        self.get_speech_timestamps, _, self.read_audio, _, _ = self.vad_utils
    
    def preprocess(self, audio_path):
        """
        Preprocess audio file: resample to 16kHz and trim silence using VAD.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file
        """
        # Load and resample audio
        wav, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        # Save to temp file for VAD processing
        temp_input_path = os.path.join(self.temp_dir, "temp_input.wav")
        torchaudio.save(temp_input_path, wav, self.sample_rate)
        
        # Run Silero VAD
        wav_vad = self.read_audio(temp_input_path, sampling_rate=self.sample_rate)
        speech_timestamps = self.get_speech_timestamps(
            wav_vad, 
            self.vad_model, 
            return_seconds=True
        )
        
        # Trim audio based on VAD
        segments = []
        for ts in speech_timestamps:
            start_sample = int(ts['start'] * self.sample_rate)
            end_sample = int(ts['end'] * self.sample_rate)
            segments.append(wav_vad[start_sample:end_sample])
        
        if segments:
            trimmed_audio = torch.cat(segments)
            trimmed_audio = trimmed_audio.unsqueeze(0)
        else:
            # No speech detected, use original audio
            trimmed_audio = wav
        
        # Save preprocessed audio
        preprocessed_path = os.path.join(self.temp_dir, "preprocessed.wav")
        torchaudio.save(preprocessed_path, trimmed_audio, self.sample_rate)
        
        return preprocessed_path
    
    def segment_audio(self, audio_path):
        """
        Segment audio into fixed-duration chunks.
        
        Args:
            audio_path: Path to preprocessed audio file
            
        Returns:
            List of paths to segmented audio files
        """
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # Calculate segment length in samples
        segment_samples = int(self.segment_duration * sr)
        
        # Calculate number of segments
        total_samples = wav.shape[1]
        num_segments = (total_samples + segment_samples - 1) // segment_samples
        
        # Create segments
        segment_paths = []
        for i in range(num_segments):
            start_idx = i * segment_samples
            end_idx = min((i + 1) * segment_samples, total_samples)
            
            segment = wav[:, start_idx:end_idx]
            
            # Pad if last segment is shorter than segment_duration
            if segment.shape[1] < segment_samples:
                padding = segment_samples - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, padding))
            
            # Save segment
            segment_path = os.path.join(self.temp_dir, f"segment_{i+1}.wav")
            torchaudio.save(segment_path, segment, sr)
            segment_paths.append(segment_path)
        
        return segment_paths
    
    def cleanup_temp_files(self):
        """Remove temporary files."""
        import glob
        temp_files = glob.glob(os.path.join(self.temp_dir, "*.wav"))
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass

