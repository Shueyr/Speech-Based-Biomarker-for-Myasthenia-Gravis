# Quick Start Guide - MG Classifier GUI

## Installation (One-time setup)

1. **Install dependencies:**
   ```bash
   cd "MG Classifier"
   pip install -r requirements.txt
   ```

2. **Verify models are present:**
   - Check that the `models/` folder contains 4 `.pth` files

## Running the Application

### Linux/Mac:
```bash
./run_gui.sh
```

### Windows:
```batch
run_gui.bat
```

### Or directly:
```bash
python mg_classifier_gui.py
```

## Quick Usage

1. **Load Audio**: Click "Browse" or "Start Recording"
2. **Select Model**: Choose from dropdown (default is Binary - Recording Split)
3. **Run**: Click "▶ Run Inference"
4. **Wait**: Processing takes 30-60 seconds
5. **View Results**: Check the "Classification Results" section

## What the Application Does

```
Your Audio → VAD Trimming → 10-sec Segments → 'ah' Detection → 
Formant Tracking → Feature Extraction → AI Classification → Result
```

## Expected Behavior

- **First run**: Downloads wav2vec2 and Silero VAD models (~500MB) - takes 2-5 minutes
- **Subsequent runs**: Much faster (~30-60 seconds per audio)
- **Progress**: All steps shown in the "Processing Status" window
- **Result**: Shows classification (Healthy/Sick or Normal/Mild/Moderate/Severe) with confidence %

## Tips for Best Results

1. **Audio Quality**: Clear speech, minimal background noise
2. **'ah' sounds**: Sustained 'ah' phonemes (like "ahhhhh") work best
3. **Duration**: At least 10-20 seconds of speech
4. **Format**: WAV files work best, 16kHz recommended

## Troubleshooting

**Problem**: "sounddevice library not installed"
**Solution**: `pip install sounddevice soundfile`

**Problem**: Slow processing
**Solution**: First run is slow (model downloads). Wait patiently.

**Problem**: "No 'ah' segments found"
**Solution**: Record audio with clear 'ah' sounds, or use a different audio file

**Problem**: Model not found
**Solution**: Ensure all 4 .pth files are in `models/` directory with exact names

## Support

See `README.md` for detailed documentation.

