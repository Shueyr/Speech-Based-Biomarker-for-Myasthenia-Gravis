# MG Classifier GUI Application

A graphical user interface application for classifying speech recordings to detect Myasthenia Gravis (MG) indicators using deep learning models.

## Attribution

This project builds upon the excellent work of the **FormantsTracker** project by Yosi Shrem, Felix Kreuk, and Joseph Keshet. The FormantsTracker component is included in this repository with modifications for MG classifier integration.

**Original FormantsTracker Repository**: https://github.com/MLSpeech/FormantsTracker

**Original Paper**: [Formant Estimation and Tracking using Probabilistic Heat-Maps](https://www.isca-speech.org/archive/pdfs/interspeech_2022/shrem22_interspeech.pdf)

**Citation**:
```
@article{shrem2022formant,
  title={Formant Estimation and Tracking using Probabilistic Heat-Maps},
  author={Shrem, Yosi and Kreuk, Felix and Keshet, Joseph},
  journal={arXiv preprint arXiv:2206.11632},
  year={2022}
}
```

**Note**: The FormantsTracker code in this repository has been modified for integration with the MG classifier pipeline. See `FormantsTracker/ATTRIBUTION.md` for details about the modifications.

## Features

- **Audio Input Options**
  - Record audio directly from microphone
  - Load existing WAV files
  
- **Multiple Classification Models**
  - Binary Classification (Healthy vs. Sick)
    - ResNet18 - Recording Split
    - ResNet18 - Speaker Split
  - Multi-class Classification (Healthy, Mild, Moderate, Severe)
    - VGG19-BN - Recording Split
    - VGG13-BN - Speaker Split

## Installation

### Prerequisites

1. Python 3.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mg-classifier.git
cd mg-classifier
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Install FormantsTracker dependencies:
```bash
cd FormantsTracker
pip install -r requirements.txt
cd ..
```

## Usage

### Starting the Application

Run the GUI application:
```bash
python mg_classifier_gui.py
```

### Using the Application

1. **Load Existing File**
     - Click "Browse"
     - Select a WAV file from your computer

2. **Choose Classification Model**
   - Use the dropdown menu to select one of the 4 available models
   - Binary models output: Healthy or Sick
   - Multi-class models output: Normal, Mild, Moderate, or Severe

3. **Run Classification**
   - Click "▶ Run Inference"
   - Wait for processing (may take 30-60 seconds)
   - View results in the "Classification Results" section

4. **Interpret Results**
   - The result shows the predicted class
   - Confidence percentage indicates model certainty
   - Green = Healthy/Normal
   - Orange = Mild
   - Red = Moderate/Severe/Sick

## Processing Pipeline

The application follows this processing pipeline:

```
Input Audio
    ↓
1. Preprocessing (Resample to 16kHz, VAD trimming)
    ↓
2. Segmentation (Split into 10-second chunks)
    ↓
3. Phoneme Detection (Detect 'ah' phonemes in each segment)
    ↓
4. Formant Tracking (Extract formant features)
    ↓
5. Segment Extraction (Extract and concatenate 'ah' segments)
    ↓
6. Model Inference (Classify using selected model)
    ↓
Output: Classification Result + Confidence
```

## Models

The application includes 4 pre-trained models located in `models/checkpoints/`:

1. **binary_split_by_recording_resnet_18.pth**
   - Architecture: ResNet18
   - Classes: 2 (Healthy, Sick)
   - Training: Split by recording

2. **binary_split_by_speaker_resnet_18.pth**
   - Architecture: ResNet18
   - Classes: 2 (Healthy, Sick)
   - Training: Split by speaker

3. **multiclass_split_by_recording_vgg_19_bn.pth**
   - Architecture: VGG19 with Batch Normalization
   - Classes: 4 (Normal, Mild, Moderate, Severe)
   - Training: Split by recording

4. **multiclass_split_by_speaker_vgg_13_bn.pth**
   - Architecture: VGG13 with Batch Normalization
   - Classes: 4 (Normal, Mild, Moderate, Severe)
   - Training: Split by speaker

## File Structure

```
MG Classifier/
├── mg_classifier_gui.py          # Main GUI application
├── audio_processor.py             # Audio preprocessing and segmentation
├── phoneme_detector.py            # Phoneme detection using wav2vec2
├── formant_processor.py           # Formant tracking integration
├── model_inference.py             # Model loading and inference
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── models/                        # Model definitions and checkpoints
│   ├── model.py                   # Model architecture definitions
│   └── checkpoints/               # Pre-trained model checkpoints
│       ├── binary_split_by_recording_resnet_18.pth
│       ├── binary_split_by_speaker_resnet_18.pth
│       ├── multiclass_split_by_recording_vgg_19_bn.pth
│       └── multiclass_split_by_speaker_vgg_13_bn.pth
└── temp/                          # Temporary files (auto-generated)
```

## Technical Details

- **Input Format**: WAV files
