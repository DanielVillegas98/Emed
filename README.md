# NeuroEdge: Ambient Care for Parkinson's Disease #emed

## The Problem
Parkinson's care has a 6-month clinical blind spot. Neurologists rely on infrequent clinic visits to adjust critical medications, while patients face daily risks of falls, freezing of gait, and voice degradation at home without continuous monitoring.

## The Solution
NeuroEdge is an edge-AI ambient monitoring system that tracks patients at home without requiring them to wear any devices. 
* **Edge Vision Tracking:** Uses a localized computer vision pipeline to monitor patient posture and instantly detect physical falls in real-time.
* **Generative Voice Triage:** Upon detecting an anomaly, the system automatically initiates a conversational assessment using LLMs to determine patient safety, pain levels, and mobility.
* **Clinical Reporting:** Instantly compiles the acute incident transcript alongside 30-day longitudinal biomarkers (e.g., medication adherence, hypophonia progression, morning gait freezing) into a structured PDF for rapid physician review.

## Tech Stack
* **Computer Vision:** OpenCV, MediaPipe Tasks Vision API (Pose Landmarker)
* **Generative AI:** Google Gemini 3.5 Flash API (`google-genai` SDK)
* **Speech & Audio:** `SpeechRecognition`, Google Text-to-Speech (`gTTS`), `pygame`
* **Data Export:** `FPDF`

## How to Run Locally

### 1. Setup the Environment
Create and activate a Python virtual environment to keep dependencies clean:
```bash
python -m venv .venv
source .venv/Scripts/activate
```
### 2. Install Dependencies
Install the required libraries for vision, LLM integration, and audio routing:
pip install opencv-python mediapipe SpeechRecognition google-genai gTTS pygame fpdf
```bash
pip install opencv-python mediapipe SpeechRecognition google-genai gTTS pygame fpdf
```
3. Configure the API Key
Export your Google Gemini API key to your local terminal session:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
4. Launch the System
Run the main tracking script. The system will automatically download the required MediaPipe .task model on its first run.

```bash
python vision_tracker.py

```
To trigger a fall: Drop your head below the red threshold line on the camera feed.

To quit: Press q while focused on the camera window, or Ctrl + C in the terminal



