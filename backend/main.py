from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
import tensorflow as tf
import string
import os
import json
from typing import List
from pydantic import BaseModel
from huggingface_hub import snapshot_download
import zipfile
import glob
from dotenv import load_dotenv
from murf import Murf

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
    # Added target_language for the translation API call
    "english": {"voice_id": "en-US-natalie", "target_language": "en", "lang_code" : 'en-US'},
    "hindi": {"voice_id": "hi-IN-ayushi", "target_language": "hi" , "lang_code" : 'hi-IN'},
    "spanish": {"voice_id": "es-ES-antonio", "target_language": "es-ES", "lang_code" : 'es-ES' },
    "french": {"voice_id": "fr-FR-celine", "target_language": "fr-FR", "lang_code" : 'fr-RF'},
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    # ... (rest of the class is unchanged)
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

# The HandSignRecognition class is unchanged.
class HandSignRecognition:
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
        self.asl_labels = list(string.ascii_uppercase) + ['space', 'del', 'nothing']
        self.model = None
        self.load_model()

    def _build_model(self, h5_model_path):
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
        print(f"⚖️  Loading weights from '{os.path.basename(h5_model_path)}'...")
        model.load_weights(h5_model_path, by_name=True)
        print("✅ Weights loaded successfully!")
        return model

    def load_model(self):
        try:
            print("📥 Downloading ASL model...")
            local_repo_path = snapshot_download(repo_id="ademaulana/CNN-ASL-Alphabet-Sign-Recognition")
            keras_model_path = os.path.join(local_repo_path, "best_cnn_asl_model.keras")
            extract_dir = local_repo_path
            with zipfile.ZipFile(keras_model_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            h5_files = glob.glob(os.path.join(extract_dir, '**', '*.h5'), recursive=True)
            weights_file_path = h5_files[0]
            print(f"✅ Found weights file: {os.path.basename(weights_file_path)}")
            self.model = self._build_model(weights_file_path)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("🎉 Model is fully loaded and compiled!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading model: {e}")
            self.model = None

    # ... (recognize_gesture and process_frame methods are unchanged)
    def recognize_gesture(self, hand_landmarks):
        try:
            if not self.model: return None
            raw_landmarks = []
            for landmark in hand_landmarks.landmark:
                raw_landmarks.extend([landmark.x, landmark.y, landmark.z])
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
        try:
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            current_letter = None
            if results.multi_hand_landmarks:
                current_letter = self.recognize_gesture(results.multi_hand_landmarks[0])
                self.last_detection_time = time.time()
            if current_letter and current_letter != self.last_detected_letter:
                self.spelled_word.append(current_letter)
                self.last_detected_letter = current_letter
            if not results.multi_hand_landmarks:
                self.last_detected_letter = None
                if self.spelled_word and (time.time() - self.last_detection_time) > self.word_completion_threshold:
                    completed_word = "".join(self.spelled_word)
                    self.spelled_word = []
                    return {"type": "word_completed", "word": completed_word}
            return {"type": "processing", "current_word": "".join(self.spelled_word)}
        except Exception as e:
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

# --- THIS IS THE UPDATED ENDPOINT ---
@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    if not MUR_API_KEY:
        raise HTTPException(status_code=500, detail="Murf API key not configured")

    voice_info = VOICE_MAP.get(request.language.lower())
    if not voice_info:
        raise HTTPException(status_code=400, detail="Unsupported language")

    try:
        murf_client = Murf(api_key=MUR_API_KEY)
        
        # Step 1: Translate the text first
        print(f"🔄 Translating '{request.text}' to {voice_info['target_language']}...")
        print(f"✅ target language: {voice_info['lang_code']}")
        print('text' , request)
        text_to_translate = []
        text_to_translate.append(request.text) 
        text_to_translate.append("we are team flag-bit") 
        translation_result = murf_client.text.translate(
            texts=text_to_translate,
            target_language=voice_info['lang_code']
        )
        print('my result' ,translation_result)
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
        # Provide a more detailed error from the API if possible
        error_message = str(e)
        if hasattr(e, 'response') and e.response:
            error_message = e.response.text
        raise HTTPException(status_code=500, detail=f"Error from Murf API: {error_message}")
# ------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "frame":
                result = recognizer.process_frame(message["data"])
                if result: await manager.send_personal_message(result, websocket)
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