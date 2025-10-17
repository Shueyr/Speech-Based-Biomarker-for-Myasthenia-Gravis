#!/usr/bin/env python3
"""
MG Classifier GUI Application
A graphical interface for classifying speech recordings using deep learning models.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
from collections import Counter

# Configuration: Set to True for per-segment results, False for majority vote
SHOW_PER_SEGMENT_RESULTS = True

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processor import AudioProcessor
from phoneme_detector import PhonemeDetector
from formant_processor import FormantProcessor
from model_inference import ModelInference
from results_manager import ResultsManager


class MGClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MG Classifier - Speech Analysis Tool")
        self.root.geometry("600x640")
        self.root.resizable(False, False)
        
        # State variables
        self.audio_file_path = tk.StringVar()
        self.selected_model = tk.StringVar()
        
        # Processing components (lazy loaded)
        self.audio_processor = None
        self.phoneme_detector = None
        self.formant_processor = None
        self.model_inference = None
        
        # Model configurations
        self.models_config = {
            "Binary - Recording Split (ResNet18)": {
                "path": "models/checkpoints/binary_split_by_recording_resnet_18.pth",
                "architecture": "resnet18",
                "num_classes": 2,
                "labels": ["Healthy", "Sick"]
            },
            "Binary - Speaker Split (ResNet18)": {
                "path": "models/checkpoints/binary_split_by_speaker_resnet_18.pth",
                "architecture": "resnet18",
                "num_classes": 2,
                "labels": ["Healthy", "Sick"]
            },
            "Multi-class - Recording Split (VGG19-BN)": {
                "path": "models/checkpoints/multiclass_split_by_recording_vgg_19_bn.pth",
                "architecture": "vgg19_bn",
                "num_classes": 4,
                "labels": ["Normal", "Mild", "Moderate", "Severe"]
            },
            "Multi-class - Speaker Split (VGG13-BN)": {
                "path": "models/checkpoints/multiclass_split_by_speaker_vgg_13_bn.pth",
                "architecture": "vgg13_bn",
                "num_classes": 4,
                "labels": ["Normal", "Mild", "Moderate", "Severe"]
            }
        }
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="MG Classifier", 
                               font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Audio Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Audio Input", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File selection
        ttk.Label(input_frame, text="Select Audio File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.audio_file_path, width=40, state='readonly').grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_file).grid(
            row=0, column=2, padx=5, pady=5)
        
        
        # Model Selection Section
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        model_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Select Classification Model:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.selected_model,
                                     values=list(self.models_config.keys()),
                                     state='readonly', width=45)
        model_dropdown.grid(row=0, column=1, padx=5, pady=5)
        model_dropdown.current(0)  # Select first model by default
        
        # Control Section
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        self.run_button = ttk.Button(control_frame, text="▶ Run Inference", 
                                     command=self.run_inference,
                                     style='Accent.TButton')
        self.run_button.grid(row=0, column=0, padx=5)
        self.run_button.config(width=20)
        
        ttk.Button(control_frame, text="Clear", command=self.clear_all).grid(
            row=0, column=1, padx=5)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Processing Status", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_text = tk.Text(progress_frame, height=8, width=70, state='disabled',
                                    wrap=tk.WORD, font=('Courier', 9))
        self.progress_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(progress_frame, orient="vertical", 
                                 command=self.progress_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.progress_text.config(yscrollcommand=scrollbar.set)
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Results text area (for multi-line results)
        self.results_text = tk.Text(results_frame, height=6, width=70, state='disabled',
                                   wrap=tk.WORD, font=('Courier', 10))
        self.results_text.grid(row=0, column=0, pady=10)
        
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", 
                                        command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        
    def browse_file(self):
        """Open file dialog to select audio file."""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.audio_file_path.set(filename)
            self.log_progress(f"Selected file: {os.path.basename(filename)}")
    
    
    def log_progress(self, message):
        """Add message to progress log."""
        self.progress_text.config(state='normal')
        self.progress_text.insert(tk.END, f"[{self.get_timestamp()}] {message}\n")
        self.progress_text.see(tk.END)
        self.progress_text.config(state='disabled')
        self.root.update_idletasks()
    
    def get_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def clear_all(self):
        """Clear all inputs and results."""
        self.audio_file_path.set("")
        self.progress_text.config(state='normal')
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state='disabled')
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "No results yet")
        self.results_text.config(state='disabled')
    
    def run_inference(self):
        """Run the complete inference pipeline."""
        # Validate inputs
        if not self.audio_file_path.get():
            messagebox.showwarning("Warning", "Please select or record an audio file first!")
            return
        
        if not self.selected_model.get():
            messagebox.showwarning("Warning", "Please select a model!")
            return
        
        if not os.path.exists(self.audio_file_path.get()):
            messagebox.showerror("Error", "Audio file not found!")
            return
        
        # Disable run button during processing
        self.run_button.config(state='disabled')
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Processing...")
        self.results_text.config(state='disabled')
        
        # Run inference in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._run_inference_thread, daemon=True)
        thread.start()
    
    def _run_inference_thread(self):
        """Run inference in a separate thread."""
        try:
            # Get model configuration
            model_config = self.models_config[self.selected_model.get()]
            audio_path = self.audio_file_path.get()
            
            # Initialize ResultsManager
            self.log_progress("Initializing results manager...")
            results_mgr = ResultsManager(audio_path)
            self.log_progress(f"✓ Results will be saved to: {results_mgr.session_dir}")
            
            # Initialize components if not already initialized
            self.log_progress("Initializing processing components...")
            
            if self.audio_processor is None:
                self.audio_processor = AudioProcessor()
                self.log_progress("✓ Audio processor ready")
            
            if self.phoneme_detector is None:
                self.phoneme_detector = PhonemeDetector()
                self.log_progress("✓ Phoneme detector ready")
            
            if self.formant_processor is None:
                self.formant_processor = FormantProcessor()
                self.log_progress("✓ Formant processor ready")
            
            if self.model_inference is None:
                self.model_inference = ModelInference()
                self.log_progress("✓ Model inference ready")
            
            # Step 1: Preprocess audio (VAD, trimming, resampling)
            self.log_progress("Step 1: Preprocessing audio...")
            processed_audio_path = self.audio_processor.preprocess(audio_path)
            self.log_progress(f"✓ Audio preprocessed")
            
            # Save preprocessed audio
            results_mgr.save_preprocessed_audio(processed_audio_path)
            
            # Step 2: Segment into 10-second chunks
            self.log_progress("Step 2: Segmenting audio into 10-second chunks...")
            segment_paths = self.audio_processor.segment_audio(processed_audio_path)
            self.log_progress(f"✓ Created {len(segment_paths)} segments")
            
            # Save segment audio files
            for i, seg_path in enumerate(segment_paths):
                results_mgr.save_segment_audio(seg_path, i + 1)
            
            # Step 3: Detect 'ah' phoneme timestamps for each segment
            self.log_progress("Step 3: Detecting 'ah' phoneme timestamps...")
            all_phoneme_segments = []
            for i, segment_path in enumerate(segment_paths):
                # Get both 'ah' timestamps and all phonemes
                phoneme_segments, all_phonemes = self.phoneme_detector.detect_phoneme(
                    segment_path, "ah", return_all_phonemes=True
                )
                all_phoneme_segments.append((segment_path, phoneme_segments))
                self.log_progress(f"  Segment {i+1}: Found {len(phoneme_segments)} 'ah' phonemes")
                
                # Save phoneme results
                results_mgr.save_phoneme_results(i + 1, phoneme_segments, all_phonemes)
            
            # Step 4: Run formant tracking and inference on each segment
            self.log_progress("Step 4: Processing each segment...")
            segment_results = []
            
            for i, (segment_path, phoneme_segments) in enumerate(all_phoneme_segments):
                segment_num = i + 1
                time_range = f"{i*10}-{min((i+1)*10, len(segment_paths)*10)}s"
                
                # Check if segment has 'ah' phonemes
                if len(phoneme_segments) == 0:
                    self.log_progress(f"  Segment {segment_num}: No 'ah' phonemes - Skipped")
                    
                    # Create visualizations even for skipped segments
                    results_mgr.create_spectrogram(segment_path, segment_num, None)
                    
                    segment_results.append({
                        'segment_number': segment_num,
                        'time_range': time_range,
                        'prediction': 'Skipped',
                        'reason': 'No /ah/ detected',
                        'ah_phonemes_count': 0
                    })
                    continue
                
                # Run formant tracking
                self.log_progress(f"  Segment {segment_num}: Running formant tracking...")
                encoder_output = self.formant_processor.process_segment(segment_path)
                
                # Save encoder output
                results_mgr.save_formant_encoder_output(segment_num, encoder_output)
                
                # Extract 'ah' segments for this segment only (with intermediate tensors)
                single_segment_data = [(encoder_output, phoneme_segments)]
                concatenated_tensor, extracted_tensor = self.formant_processor.extract_and_concatenate_ah_segments(
                    single_segment_data, return_intermediate=True
                )
                
                # Save intermediate and final tensors
                results_mgr.save_ah_extracted_tensor(segment_num, extracted_tensor)
                results_mgr.save_ah_normalized_tensor(segment_num, concatenated_tensor)
                
                # Create visualizations
                self.log_progress(f"  Segment {segment_num}: Creating visualizations...")
                results_mgr.create_spectrogram(segment_path, segment_num, phoneme_segments)
                results_mgr.create_formant_overlay(segment_path, segment_num, encoder_output, phoneme_segments)
                results_mgr.create_ah_regions_plot(segment_path, segment_num, phoneme_segments)
                results_mgr.create_model_input_heatmap(segment_num, concatenated_tensor)
                
                # Run model inference
                self.log_progress(f"  Segment {segment_num}: Running inference...")
                prediction, confidence, label = self.model_inference.predict(
                    concatenated_tensor,
                    model_config['path'],
                    model_config['architecture'],
                    model_config['num_classes'],
                    model_config['labels']
                )
                
                self.log_progress(f"  Segment {segment_num}: {label}")
                
                # Store result with probabilities
                result_data = {
                    'segment_number': segment_num,
                    'time_range': time_range,
                    'prediction': label,
                    'reason': None,
                    'ah_phonemes_count': len(phoneme_segments),
                    'class_probabilities': {
                        model_config['labels'][j]: float(confidence) if j == prediction else 0.0
                        for j in range(model_config['num_classes'])
                    }
                }
                segment_results.append(result_data)
                
                # Save individual segment prediction
                results_mgr.save_segment_prediction(result_data)
            
            self.log_progress(f"✓ Processing complete!")
            
            # Save final comprehensive results
            self.log_progress("Saving final results...")
            results_mgr.save_final_results(segment_results, self.selected_model.get())
            
            # Update results in GUI
            self.root.after(0, self._update_results, segment_results, model_config['labels'])
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.log_progress(f"❌ {error_msg}")
            self.root.after(0, messagebox.showerror, "Error", error_msg)
            self.root.after(0, self._reset_results)
        finally:
            # Re-enable run button
            self.root.after(0, lambda: self.run_button.config(state='normal'))
    
    def _update_results(self, segment_results, class_labels):
        """Update results display based on SHOW_PER_SEGMENT_RESULTS setting."""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        
        if SHOW_PER_SEGMENT_RESULTS:
            # Display per-segment results
            for result in segment_results:
                seg_num = result['segment_number']
                time_range = result['time_range']
                prediction = result['prediction']
                
                if result['reason']:
                    line = f"Segment {seg_num} ({time_range}): {result['reason']}\n"
                else:
                    line = f"Segment {seg_num} ({time_range}): {prediction}\n"
                
                self.results_text.insert(tk.END, line)
        else:
            # Display majority vote result
            valid_predictions = [r['prediction'] for r in segment_results if r['reason'] is None]
            
            if not valid_predictions:
                self.results_text.insert(tk.END, "No valid segments to classify\n")
            else:
                # Count predictions
                vote_counts = Counter(valid_predictions)
                majority_class = vote_counts.most_common(1)[0][0]
                majority_count = vote_counts[majority_class]
                total_valid = len(valid_predictions)
                
                # Display result
                self.results_text.insert(tk.END, f"Result: {majority_class}\n")
                self.results_text.insert(tk.END, f"(Majority vote: {majority_count}/{total_valid} segments)\n")
                
                # Show vote breakdown
                self.results_text.insert(tk.END, "\nVote breakdown:\n")
                for class_name, count in vote_counts.most_common():
                    self.results_text.insert(tk.END, f"  {class_name}: {count}\n")
        
        self.results_text.config(state='disabled')
    
    def _reset_results(self):
        """Reset results display."""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Processing failed")
        self.results_text.config(state='disabled')


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = MGClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

