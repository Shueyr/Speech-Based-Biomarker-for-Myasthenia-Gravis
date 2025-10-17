"""
Phoneme Detector Module
Detects phoneme timestamps in audio using wav2vec2 model.
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class PhonemeDetector:
    """Detects specific phonemes in audio files."""
    
    def __init__(self, model_name="vitouphy/wav2vec2-xls-r-300m-phoneme"):
        """
        Initialize phoneme detector.
        
        Args:
            model_name: HuggingFace model name for phoneme recognition
        """
        print(f"Loading phoneme detection model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        print("Phoneme detection model loaded successfully")
    
    def detect_phoneme(self, audio_path, target_phoneme="ah", return_all_phonemes=False):
        """
        Detect timestamps of a specific phoneme in audio file.
        
        Args:
            audio_path: Path to audio file
            target_phoneme: Phoneme to detect (default: "ah")
            return_all_phonemes: If True, return all phonemes; if False, only target phoneme
            
        Returns:
            If return_all_phonemes=False: List of (start_time, end_time) tuples for target phoneme
            If return_all_phonemes=True: Tuple of (target_phoneme_list, all_phonemes_list)
        """
        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        
        # Convert to mono and numpy
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform[0]
        
        waveform_np = waveform.numpy()
        
        # Process with wav2vec2
        inputs = self.processor(waveform_np, return_tensors="pt", sampling_rate=16000)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get predicted phoneme IDs
        pred_ids = torch.argmax(logits, dim=-1)[0]
        
        # Calculate time stride (frames per second)
        audio_duration = len(waveform_np) / 16000
        time_stride = logits.shape[1] / audio_duration
        
        # Extract phoneme segments
        phoneme_segments = []
        all_phonemes = [] if return_all_phonemes else None
        prev = None
        start_frame = 0
        
        for i, phoneme_id in enumerate(pred_ids):
            if phoneme_id != prev:
                if prev is not None:
                    phoneme = self.processor.tokenizer.convert_ids_to_tokens(prev.item())
                    start_time = start_frame / time_stride
                    end_time = i / time_stride
                    
                    # Add to all phonemes list if requested
                    if return_all_phonemes:
                        all_phonemes.append({
                            "phoneme": phoneme,
                            "start": float(start_time),
                            "end": float(end_time)
                        })
                    
                    # Add to target phoneme list
                    if phoneme == target_phoneme:
                        phoneme_segments.append((start_time, end_time))
                
                start_frame = i
                prev = phoneme_id
        
        # Handle last phoneme
        if prev is not None:
            phoneme = self.processor.tokenizer.convert_ids_to_tokens(prev.item())
            start_time = start_frame / time_stride
            end_time = len(pred_ids) / time_stride
            
            if return_all_phonemes:
                all_phonemes.append({
                    "phoneme": phoneme,
                    "start": float(start_time),
                    "end": float(end_time)
                })
            
            if phoneme == target_phoneme:
                phoneme_segments.append((start_time, end_time))
        
        if return_all_phonemes:
            return phoneme_segments, all_phonemes
        else:
            return phoneme_segments

