# Text2Speech2Spectrogram-App

A Python application that converts text to speech and displays the spectrogram visualization alongside the audio waveform.

## Screenshot

![Application Screenshot](https://raw.githubusercontent.com/bemoregt/Text2Speech2Spectrogram-App/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-07-27_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.41.12.png)

## Overview

This application uses PyTorch and TorchAudio to convert text into speech using the Tacotron2 and WaveRNN models. It provides a simple GUI interface built with Tkinter that allows users to:

1. Enter text to be converted to speech
2. Generate the speech audio and visualize its spectrogram
3. Play the generated audio by clicking on the graph

## Features

- Text-to-speech conversion using pre-trained Tacotron2 and WaveRNN models
- Spectrogram visualization of the generated speech
- Waveform visualization of the audio
- Interactive playback by clicking on the visualization
- Simple, user-friendly interface

## Requirements

- Python 3.6+
- PyTorch
- TorchAudio
- Tkinter
- Pygame
- Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/bemoregt/Text2Speech2Spectrogram-App.git
cd Text2Speech2Spectrogram-App

# Install dependencies
pip install torch torchaudio matplotlib pygame
```

## Usage

1. Run the application:
```bash
python text2speech.py
```

2. Enter text in the input field
3. Click the "Generate" button to convert text to speech and display the spectrogram
4. Click anywhere on the displayed graph to play the generated audio

## How It Works

1. The application uses the TACOTRON2_WAVERNN_PHONE_LJSPEECH pipeline from TorchAudio
2. Text is processed and converted to phonemes
3. Tacotron2 generates a mel-spectrogram from the phonemes
4. WaveRNN vocoder converts the spectrogram to an audio waveform
5. The generated audio is saved as "output.wav" and can be played through the interface

## Technical Details

- The top subplot shows the spectrogram of the generated speech
- The bottom subplot shows the waveform of the audio
- Audio playback is handled by Pygame's mixer module
- GPU acceleration is used if available, otherwise falls back to CPU

## License

MIT

## Author

@bemoregt