"""
Formant Processor Module
Handles formant tracking and extraction of 'ah' phoneme segments.
"""

import torch
import torch.nn.functional as F
import os
import sys
from argparse import Namespace
from omegaconf import OmegaConf
import numpy as np


class FormantProcessor:
    """Processes audio to extract formant features."""
    
    def __init__(self):
        """Initialize formant processor."""
        # Add FormantsTracker to path
        self.original_path = os.path.dirname(os.path.abspath(__file__))
        self.formants_path = os.path.join(self.original_path, "FormantsTracker")
        
        if self.formants_path not in sys.path:
            sys.path.insert(0, self.formants_path)
        
        # Import formant tracking modules
        from solver import Solver
        
        self.Solver = Solver
        
        # Load configuration
        self.config = self._load_config()
        
        print("Formant processor initialized")
    
    def _load_config(self):
        """Load formant tracker configuration."""
        config_path = os.path.join(self.formants_path, "conf", "config.yaml")
        
        cfg = OmegaConf.load(config_path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Convert to Namespace
        cfg = Namespace(**cfg_dict)
        cfg.device = torch.device("cuda" if (torch.cuda.is_available() and cfg.is_cuda) else "cpu")
        
        return cfg
    
    def process_segment(self, audio_path):
        """
        Run formant tracking on a single audio segment.
        
        Args:
            audio_path: Path to audio segment
            
        Returns:
            encoder_output: Tensor of shape [F, T] (frequency x time)
        """
        # Update config with current audio path
        temp_dir = os.path.join(self.original_path, "temp", "formant_input")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy audio to temp directory (FormantsTracker expects directory input)
        import shutil
        temp_audio_path = os.path.join(temp_dir, os.path.basename(audio_path))
        shutil.copy(audio_path, temp_audio_path)
        
        self.config.test_dir = temp_dir
        self.config.predictions_dir = os.path.join(self.original_path, "temp", "formant_output")
        os.makedirs(self.config.predictions_dir, exist_ok=True)
        
        # Change to formants directory
        original_cwd = os.getcwd()
        os.chdir(self.formants_path)
        
        try:
            # Run formant tracking
            solver = self.Solver(self.config)
            encoder_outputs, pred_formants_list, fnames = solver.test()
            
            # Return first (and only) result
            if len(encoder_outputs) > 0:
                encoder_output = encoder_outputs[0]  # Shape: [F, T]
                
                return encoder_output
            else:
                raise RuntimeError("Formant tracking returned no results")
        
        finally:
            # Return to original directory
            os.chdir(original_cwd)
            
            # Cleanup temp files
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def extract_and_concatenate_ah_segments(self, formant_data_list, fixed_len=64):
        """
        Extract 'ah' phoneme segments from formant data and concatenate them.
        
        Args:
            formant_data_list: List of tuples (encoder_output, phoneme_segments)
            fixed_len: Target length in time frames (default: 64)
            
        Returns:
            Tensor of shape [1, 257, 64] ready for model input
        """
        all_segments = []
        
        for encoder_output, phoneme_segments in formant_data_list:
            # encoder_output shape: [F, T] where F=257 (frequency bins)
            # phoneme_segments: list of (start_time, end_time) in seconds
            
            # Convert phoneme timestamps to frame indices
            # Formant tracker uses 10ms frames (100 frames per second)
            for start_time, end_time in phoneme_segments:
                start_frame = int(start_time * 100)  # 100 frames per second
                end_frame = int(end_time * 100)
                
                # Check bounds
                if start_frame >= encoder_output.shape[1]:
                    continue
                end_frame = min(end_frame, encoder_output.shape[1])
                
                if end_frame > start_frame:
                    segment = encoder_output[:, start_frame:end_frame]
                    all_segments.append(segment)
        
        if not all_segments:
            # No 'ah' segments found, return zeros
            print("Warning: No 'ah' segments found, using zeros")
            return torch.zeros((1, 257, fixed_len), dtype=torch.float32)
        
        # Concatenate all segments
        concatenated = torch.cat(all_segments, dim=1)  # Shape: [257, total_frames]
        
        # Pad or crop to fixed_len
        cur_len = concatenated.shape[1]
        if cur_len < fixed_len:
            # Pad with zeros
            pad_width = fixed_len - cur_len
            concatenated = F.pad(concatenated, (0, pad_width))
        else:
            # Crop to fixed_len
            concatenated = concatenated[:, :fixed_len]
        
        # Add batch dimension: [1, 257, 64]
        concatenated = concatenated.unsqueeze(0)
        
        return concatenated

