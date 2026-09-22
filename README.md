# 🪔 Indian Festival Recognition System

A real-time Computer Vision and AI Web Application that detects Indian festivals (like Diwali, Holi, Pongal) through a live webcam feed and automatically plays corresponding traditional songs or Aartis in a responsive web dashboard.

## ✨ Features
- **Real-Time Detection**: Captures live webcam feed and runs inference locally using OpenCV and TensorFlow.
- **Advanced Deep Learning**: Uses a custom-trained `MobileNetV2` model with Transfer Learning and aggressive Data Augmentation for robust accuracy.
- **Responsive Web Dashboard**: A beautifully designed UI built with Flask and Tailwind CSS that displays live video streams and real-time confidence metrics.
- **Smart Audio Playback**: Automatically maps detected festivals to MP3 tracks in the `static/audio/` folder based on file name matching. 
- **Non-blocking Architecture**: Inference runs in separate threads via Flask MJPEG streaming so the browser UI remains buttery smooth.

## 📁 Project Structure
```
IFS_System/
├── app.py                      # Main Flask Web Server & OpenCV streaming logic
├── train_model.py              # Robust Transfer Learning script (Keras 3)
├── festival_model.keras        # Pre-trained MobileNetV2 model
├── festival_model.labels.json  # Auto-generated label mappings
├── templates/
│   └── index.html              # Tailwind CSS Responsive Dashboard
├── static/
│   └── audio/                  # Drop your MP3 files here (e.g., Happy_Diwali.mp3)
└── requirements.txt            # Python dependencies
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Karankumawa/IFS_System.git
   cd IFS_System
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎵 How to configure Audio
The system uses **Auto-Mapping**. 
1. Place your MP3 files inside the `static/audio/` folder.
2. Make sure the festival name is somewhere in the file name. 
   - *Example: For the `DIWALI` class, name your file `Happy_Diwali.mp3` or `diwali_song.mp3`.*
3. The backend will automatically detect the name and map it to the UI.

## 🏃‍♂️ Usage

### 1. (Optional) Retrain the Model
If you want to train the model on your own custom dataset, place your folders inside `dataset/festivals_classification/train` and run:
```bash
python train_model.py
```
*This will apply data augmentation, train the MobileNetV2 model, save it as `festival_model.keras`, and export the `labels.json`.*

### 2. Start the Dashboard
Run the Flask server:
```bash
python app.py
```
Open your web browser and go to:
**http://127.0.0.1:5000**

> **Note**: Modern browsers block auto-playing audio for security reasons. After opening the dashboard, simply click anywhere on the page once to allow the audio player to start reacting to live detections.

## 🛠 Technologies Used
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras (MobileNetV2)
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
