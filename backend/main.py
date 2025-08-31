import json
import os
import glob
import zipfile
import time
import base64
from typing import List
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from murf import Murf
import string

# Load environment variables from .env file
load_dotenv()
MUR_API_KEY = os.getenv("MUR_API_KEY")

app = FastAPI(title="Hand Sign Recognition API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Voice and Language configuration
VOICE_MAP = {
    "english": {"voice_id": "en-US-natalie", "target_language": "en", "lang_code": 'en-US'},
    "hindi": {"voice_id": "hi-IN-ayushi", "target_language": "hi", "lang_code": 'hi-IN'},
    "spanish": {"voice_id": "es-ES-antonio", "target_language": "es", "lang_code": 'es-ES'},
    "french": {"voice_id": "fr-FR-celine", "target_language": "fr", "lang_code": 'fr-FR'},
}

class ConnectionManager:
    """
    Manages active WebSocket connections.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

class HandSignRecognition:
    """
    A class to recognize hand gestures using a pre-trained model and manage word building.
    """
    def __init__(self, language="english"):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.language = language.lower()
        self.voice_id = VOICE_MAP.get(self.language, VOICE_MAP["english"])["voice_id"]
        self.spelled_word = []
        self.last_detected_letter = None
        self.last_detection_time = time.time()
        self.word_completion_threshold = 2.0
        # Labels for the model output
        self.asl_labels = list(string.ascii_uppercase) + ['space', 'del', 'nothing']
        self.model = None
        self.model_ready = False
        self.load_model()

    def _build_model(self, model_path):
        """Builds and loads weights into the TensorFlow model from a specified path."""
        print("🏗️  Manually building model architecture...")
        inputs = tf.keras.layers.Input(shape=(63, 1), name="input")
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', name='conv1d')(inputs)
        x = tf.keras.layers.MaxPooling1D(2, name='max_pooling1d')(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', name='conv1d_1')(x)
        x = tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_1')(x)
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense')(x)
        x = tf.keras.layers.Dropout(0.4, name='dropout')(x)
        outputs = tf.keras.layers.Dense(29, activation='softmax', name='dense_1')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_model_rebuilt")
        print(f"⚖️  Loading weights from '{os.path.basename(model_path)}'...")
        model.load_weights(model_path, by_name=True)
        print("✅ Weights loaded successfully!")
        return model

    def load_model(self):
        """
        Downloads and loads the ASL recognition model from Hugging Face Hub.
        """
        if self.model_ready:
            print("Model is already loaded. Skipping download.")
            return

        try:
            print("📥 Downloading ASL model...")
            local_repo_path = snapshot_download(repo_id="ademaulana/CNN-ASL-Alphabet-Sign-Recognition")
            h5_files = glob.glob(os.path.join(local_repo_path, '**', '*.h5'), recursive=True)
            
            if not h5_files:
                raise FileNotFoundError("No .h5 weights file found after download.")
                
            weights_file_path = h5_files[0]
            print(f"✅ Found weights file: {os.path.basename(weights_file_path)}")
            
            self.model = self._build_model(weights_file_path)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model_ready = True
            print("🎉 Model is fully loaded and compiled!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading model: {e}")
            self.model = None
            self.model_ready = False

    def recognize_gesture(self, hand_landmarks):
        """
        Recognizes a gesture using the loaded TensorFlow model.
        """
        try:
            if not self.model_ready: return None
            
            raw_landmarks = []
            for landmark in hand_landmarks.landmark:
                raw_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
            # Normalize landmarks
            relative_landmarks = [val - raw_landmarks[i % 3] for i, val in enumerate(raw_landmarks)]
            max_abs_val = max(map(abs, relative_landmarks))
            if max_abs_val == 0: return None
            normalized_landmarks = [val / max_abs_val for val in relative_landmarks]
            
            input_data = np.array(normalized_landmarks).reshape(1, 63, 1)
            
            predictions = self.model.predict(input_data, verbose=0)
            predicted_index = np.argmax(predictions)
            confidence = predictions[0][predicted_index]
            predicted_letter = self.asl_labels[predicted_index]
            
            print(f"🎯 Prediction: {predicted_letter} (confidence: {confidence:.3f})")
            
            return predicted_letter if confidence > 0.70 else None
        except Exception as e:
            print(f"❌ Error in recognize_gesture: {e}")
            return None

    def process_frame(self, frame_data):
        """
        Processes a single video frame to detect gestures and build a word.
        """
        try:
            # Decode the base64 string to an image
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convert the image to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image to find hand landmarks
            results = self.hands.process(img_rgb)
            
            current_sign = None
            if results.multi_hand_landmarks:
                # If a hand is detected, recognize the gesture
                current_sign = self.recognize_gesture(results.multi_hand_landmarks[0])
                self.last_detection_time = time.time()
                
            if current_sign and current_sign != self.last_detected_letter:
                # Handle special signs 'space' and 'del'
                if current_sign == 'space':
                    self.spelled_word.append(' ')
                elif current_sign == 'del':
                    if self.spelled_word:
                        self.spelled_word.pop()
                elif current_sign != 'nothing':
                    self.spelled_word.append(current_sign)
                
                self.last_detected_letter = current_sign
                
            if not results.multi_hand_landmarks:
                # If no hand is detected, reset the last detected letter state
                self.last_detected_letter = None
                
                # Check for word completion based on time since last detection
                if self.spelled_word and (time.time() - self.last_detection_time) > self.word_completion_threshold:
                    completed_word = "".join(self.spelled_word).strip()
                    self.spelled_word = []
                    return {"type": "word_completed", "word": completed_word}
                    
            return {"type": "processing", "current_word": "".join(self.spelled_word)}
            
        except Exception as e:
            # Return an error message if processing fails
            return {"type": "error", "message": str(e)}

recognizer = HandSignRecognition()

class LanguageRequest(BaseModel):
    language: str

class SpeechRequest(BaseModel):
    text: str
    language: str

@app.post("/set_language")
async def set_language(request: LanguageRequest):
    global recognizer
    if request.language.lower() in VOICE_MAP:
        recognizer = HandSignRecognition(request.language)
        return {"status": "success", "language": request.language}
    raise HTTPException(status_code=400, detail="Language not supported")

@app.get("/languages")
async def get_languages():
    return list(VOICE_MAP.keys())

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    if not MUR_API_KEY:
        raise HTTPException(status_code=500, detail="Murf API key not configured")

    voice_info = VOICE_MAP.get(request.language.lower())
    if not voice_info:
        raise HTTPException(status_code=400, detail="Unsupported language")

    try:
        murf_client = Murf(api_key=MUR_API_KEY)
        
        # Step 1: Translate the text
        
        print(f"🔄 Translating '{request.text}' to {voice_info['lang_code']}...")
        translation_result = murf_client.text.translate(
            texts=[request.text], # Only translate the user's text
            target_language=voice_info['lang_code']
        )
        translated_text = translation_result.translations[0].translated_text
        print(f"✅ Translated text: '{translated_text}'")

        # Step 2: Generate speech from the translated text
        print(f"🗣️ Generating speech for '{translated_text}'...")
        response_iterator = murf_client.text_to_speech.stream(
            text=translated_text,
            voice_id=voice_info["voice_id"]
        )
        return StreamingResponse(response_iterator, media_type="audio/mpeg")

    except Exception as e:
        error_message = str(e)
        if hasattr(e, 'response') and e.response:
            error_message = e.response.text
        raise HTTPException(status_code=500, detail=f"Error from Murf API: {error_message}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "frame":
                result = recognizer.process_frame(message["data"])
                if result:
                    await manager.send_personal_message(result, websocket)
            elif message["type"] == "reset":
                recognizer.spelled_word = []
                recognizer.last_detected_letter = None
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return {"message": "Hand Sign Recognition API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
