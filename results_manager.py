"""
Results Manager Module
Handles saving all intermediate results and visualizations.
"""

import os
import json
import shutil
import torch
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torchaudio


class ResultsManager:
    """Manages saving of all intermediate results and visualizations."""
    
    def __init__(self, original_audio_path, output_base_dir="results"):
        """
        Initialize results manager.
        
        Args:
            original_audio_path: Path to the original input audio file
            output_base_dir: Base directory for saving results
        """
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.splitext(os.path.basename(original_audio_path))[0]
        self.session_dir = os.path.join(output_base_dir, f"{timestamp}_{original_filename}")
        
        # Create subdirectories
        self.audio_dir = os.path.join(self.session_dir, "audio", "segments")
        self.phonemes_dir = os.path.join(self.session_dir, "phonemes")
        self.formants_dir = os.path.join(self.session_dir, "formants")
        self.viz_dir = os.path.join(self.session_dir, "visualizations")
        self.results_dir = os.path.join(self.session_dir, "predictions")
        
        for directory in [self.audio_dir, self.phonemes_dir, self.formants_dir, 
                         self.viz_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print(f"Results will be saved to: {self.session_dir}")
        
        # Copy original audio
        original_dest = os.path.join(self.session_dir, "audio", "01_original.wav")
        shutil.copy2(original_audio_path, original_dest)
        print(f"✓ Saved original audio")
    
    def save_preprocessed_audio(self, audio_path):
        """Save preprocessed audio file."""
        dest = os.path.join(self.session_dir, "audio", "02_preprocessed.wav")
        shutil.copy2(audio_path, dest)
        print(f"✓ Saved preprocessed audio")
    
    def save_segment_audio(self, segment_path, segment_num):
        """Save audio segment."""
        dest = os.path.join(self.audio_dir, f"segment_{segment_num}.wav")
        shutil.copy2(segment_path, dest)
    
    def save_phoneme_results(self, segment_num, ah_timestamps, all_phonemes=None):
        """
        Save phoneme detection results.
        
        Args:
            segment_num: Segment number
            ah_timestamps: List of (start, end) tuples for 'ah' phonemes
            all_phonemes: Optional list of all detected phonemes with timestamps
        """
        # Save 'ah' only timestamps
        ah_data = {
            "segment": segment_num,
            "ah_phonemes": [
                {"start": float(start), "end": float(end)}
                for start, end in ah_timestamps
            ],
            "count": len(ah_timestamps)
        }
        
        ah_file = os.path.join(self.phonemes_dir, f"segment_{segment_num}_ah_only.json")
        with open(ah_file, 'w') as f:
            json.dump(ah_data, f, indent=2)
        
        # Save all phonemes if provided
        if all_phonemes:
            all_file = os.path.join(self.phonemes_dir, f"segment_{segment_num}_all_phonemes.json")
            with open(all_file, 'w') as f:
                json.dump(all_phonemes, f, indent=2)
    
    def save_formant_encoder_output(self, segment_num, encoder_output):
        """Save raw encoder output tensor."""
        path = os.path.join(self.formants_dir, f"segment_{segment_num}_encoder.pt")
        torch.save(encoder_output, path)
    
    def save_ah_extracted_tensor(self, segment_num, ah_tensor):
        """Save extracted 'ah' regions tensor."""
        path = os.path.join(self.formants_dir, f"segment_{segment_num}_ah_extracted.pt")
        torch.save(ah_tensor, path)
    
    def save_ah_normalized_tensor(self, segment_num, normalized_tensor):
        """Save final normalized [1, 257, 64] tensor fed to model."""
        path = os.path.join(self.formants_dir, f"segment_{segment_num}_ah_normalized.pt")
        torch.save(normalized_tensor, path)
    
    def create_spectrogram(self, audio_path, segment_num, ah_timestamps=None):
        """
        Create and save spectrogram visualization.
        
        Args:
            audio_path: Path to audio file
            segment_num: Segment number
            ah_timestamps: Optional list of (start, end) tuples to highlight
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Create spectrogram
        spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            hop_length=256,
            power=2.0
        )
        spec = spectrogram_transform(waveform)
        spec_db = 10 * torch.log10(spec + 1e-10)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(
            spec_db[0].numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, waveform.shape[1]/sr, 0, sr/2]
        )
        
        # Highlight 'ah' regions if provided
        if ah_timestamps:
            for start, end in ah_timestamps:
                ax.axvspan(start, end, alpha=0.3, color='red', label='ah phoneme')
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram - Segment {segment_num}')
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Save
        path = os.path.join(self.viz_dir, f"segment_{segment_num}_spectrogram.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def create_formant_overlay(self, audio_path, segment_num, encoder_output, ah_timestamps=None):
        """
        Create spectrogram with formant features overlay.
        
        Args:
            audio_path: Path to audio file
            segment_num: Segment number
            encoder_output: Formant encoder output [257, T]
            ah_timestamps: Optional 'ah' timestamps to highlight
        """
        # Load audio for duration
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sr
        
        # Plot encoder output as heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Encoder output represents formant features over time
        time_frames = encoder_output.shape[1]
        time_axis = np.linspace(0, duration, time_frames)
        
        im = ax.imshow(
            encoder_output.cpu().numpy(),
            aspect='auto',
            origin='lower',
            cmap='plasma',
            extent=[0, duration, 0, 257],
            interpolation='bilinear'
        )
        
        # Highlight 'ah' regions
        if ah_timestamps:
            for start, end in ah_timestamps:
                ax.axvspan(start, end, alpha=0.2, color='lime', label='ah phoneme')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Formant Feature Dimension')
        ax.set_title(f'Formant Features with /ah/ Overlay - Segment {segment_num}')
        plt.colorbar(im, ax=ax, label='Feature Intensity')
        
        path = os.path.join(self.viz_dir, f"segment_{segment_num}_formants_overlay.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def create_ah_regions_plot(self, audio_path, segment_num, ah_timestamps):
        """
        Create plot showing just the 'ah' regions.
        
        Args:
            audio_path: Path to audio file
            segment_num: Segment number
            ah_timestamps: List of (start, end) tuples
        """
        if not ah_timestamps:
            return
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform[0]
        
        duration = waveform.shape[0] / sr
        time_axis = np.linspace(0, duration, waveform.shape[0])
        
        # Plot waveform with highlighted regions
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, waveform.numpy(), linewidth=0.5, color='blue', alpha=0.7)
        
        for i, (start, end) in enumerate(ah_timestamps):
            ax.axvspan(start, end, alpha=0.3, color='red', 
                      label='ah phoneme' if i == 0 else '')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform with /ah/ Phoneme Regions - Segment {segment_num}')
        if ah_timestamps:
            ax.legend()
        
        path = os.path.join(self.viz_dir, f"segment_{segment_num}_ah_regions.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def create_model_input_heatmap(self, segment_num, normalized_tensor):
        """
        Create heatmap of the [1, 257, 64] tensor fed to model.
        
        Args:
            segment_num: Segment number
            normalized_tensor: The [1, 257, 64] or [257, 64] tensor
        """
        # Remove batch dimension if present
        if normalized_tensor.dim() == 3:
            tensor_2d = normalized_tensor[0]
        else:
            tensor_2d = normalized_tensor
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            tensor_2d.cpu().numpy(),
            aspect='auto',
            origin='lower',
            cmap='hot',
            interpolation='bilinear'
        )
        
        ax.set_xlabel('Time Frames (normalized to 64)')
        ax.set_ylabel('Formant Feature Dimension (257)')
        ax.set_title(f'Model Input Tensor - Segment {segment_num}')
        plt.colorbar(im, ax=ax, label='Feature Value')
        
        path = os.path.join(self.viz_dir, f"segment_{segment_num}_model_input_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def save_segment_prediction(self, segment_results):
        """
        Save prediction for a single segment.
        
        Args:
            segment_results: Dict with segment prediction info
        """
        path = os.path.join(self.results_dir, f"segment_{segment_results['segment_number']}_prediction.json")
        with open(path, 'w') as f:
            json.dump(segment_results, f, indent=2)
    
    def save_final_results(self, all_segment_results, model_name):
        """
        Save comprehensive final results.
        
        Args:
            all_segment_results: List of all segment prediction dicts
            model_name: Name of the model used
        """
        # Save per-segment results JSON
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "segments": all_segment_results
        }
        
        json_path = os.path.join(self.results_dir, "per_segment_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create summary report
        self._create_summary_report(all_segment_results, model_name)
        
        print(f"✓ All results saved to: {self.session_dir}")
    
    def _create_summary_report(self, all_segment_results, model_name):
        """Create human-readable summary report."""
        report_path = os.path.join(self.results_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MG CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_name}\n\n")
            
            f.write("SEGMENTATION & PHONEME DETECTION\n")
            f.write("-" * 60 + "\n")
            valid_count = 0
            for result in all_segment_results:
                seg_num = result['segment_number']
                time_range = result['time_range']
                
                if result.get('reason'):
                    f.write(f"Segment {seg_num} ({time_range}): {result['reason']}\n")
                else:
                    ah_count = result.get('ah_phonemes_count', 0)
                    f.write(f"Segment {seg_num} ({time_range}): {ah_count} 'ah' phonemes detected\n")
                    valid_count += 1
            
            f.write(f"\nTotal segments: {len(all_segment_results)}\n")
            f.write(f"Valid segments: {valid_count}\n\n")
            
            f.write("CLASSIFICATION RESULTS\n")
            f.write("-" * 60 + "\n")
            for result in all_segment_results:
                seg_num = result['segment_number']
                time_range = result['time_range']
                prediction = result['prediction']
                
                if result.get('reason'):
                    f.write(f"Segment {seg_num} ({time_range}): SKIPPED ({result['reason']})\n")
                else:
                    f.write(f"Segment {seg_num} ({time_range}): {prediction}\n")
            
            # Calculate majority vote
            valid_predictions = [r['prediction'] for r in all_segment_results if not r.get('reason')]
            
            if valid_predictions:
                from collections import Counter
                vote_counts = Counter(valid_predictions)
                majority_class = vote_counts.most_common(1)[0][0]
                
                f.write("\nFINAL RESULT\n")
                f.write("-" * 60 + "\n")
                f.write(f"Classification: {majority_class}\n")
                f.write(f"Basis: Majority vote ({vote_counts[majority_class]}/{len(valid_predictions)} segments)\n\n")
                f.write("Vote Breakdown:\n")
                for class_name, count in vote_counts.most_common():
                    f.write(f"  - {class_name}: {count} segment(s)\n")
            
            f.write("\n" + "=" * 60 + "\n")
