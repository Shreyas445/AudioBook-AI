# 📚 NexGen AI Audiobook Converter

Turn any text file or PDF into an immersive audiobook with distinct character voices via **Kokoro** and ambient background music via **AudioLDM 2**.

## 🚀 Features

- **Character Voices**: Automatically assigns voices to characters using `Kokoro-82M`.
- **Atmospheric Music**: Generates background soundtracks (Calm, Tense, Mystery, etc.) using `AudioLDM 2` and loops them intelligently.
- **Smart Stitching**: Mixes speech and music with auto-ducking (music fades when people speak).
- **GPU Accelerated**: Runs 10x faster on NVIDIA GPUs (CUDA 12.1).
- **GUI Interface**: Simple drag-and-drop web interface.

---

## 🛠️ Installation

1.  **Install Python**: Ensure Python 3.10+ is installed.
2.  **Install Dependencies**:
    Open the terminal in this folder and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install Ollama**:
    - Download from [ollama.com](https://ollama.com).
    - Run `ollama pull llama3`.
    - Run `ollama serve` in a separate terminal.

---

## 🎮 How to Run

### Option 1: The GUI (Recommended)
1.  Double-click `run_gui.bat`.
2.  A browser window will open.
3.  **Drag & Drop** your `.txt` or `.pdf` file.
4.  Click **"Convert"** and listen!

### Option 2: Command Line
Drag your text file onto `run.bat`, or run:
```bash
python main.py my_book.txt
```

---

## 📂 Output
- **Audiobook**: Saved to `output/audiobook.mp3`.
- **Music Library**: Generated themes are stored in `output/music_library` and reused (saving time/space).

## ⚠️ Troubleshooting

- **Disk Space**: The AI models need ~7GB of space. If the app crashes, check your C: drive space.
- **CUDA Warning**: If you see "CUDA not available", your generation will be slow (CPU mode). Ensure you have NVIDIA Drivers installed.
- **Ollama Error**: Ensure `ollama serve` is running in the background.
