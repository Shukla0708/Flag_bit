# Real-Time Hand Sign Recognition with Multi-Language Speech

This project is a sophisticated web application that translates American Sign Language (ASL) gestures into text in real-time and then converts that text into audible speech in multiple languages. It leverages a powerful machine learning model, real-time communication with WebSockets, and a modern web interface to create an interactive and intuitive experience.

The application captures video from the user's webcam, processes it on a Python backend, and displays the recognized text on a React frontend, bridging the gap between sign language and spoken language.

## ✨ Features

* **Real-Time Gesture Recognition**: Uses a TensorFlow CNN model to recognize ASL alphabet signs from a live video feed.
* **Dynamic Text Assembly**: Intelligently detects individual signs and assembles them into words and sentences.
* **Multi-Language Text-to-Speech**: Translates the recognized English text and generates high-quality speech in multiple languages (English, Hindi, Spanish, French) using the Murf AI API.
* **Interactive Web Interface**: A user-friendly frontend built with React and styled with Tailwind CSS provides camera controls, language selection, and a clear display for the recognized text.
* **WebSocket Communication**: Ensures low-latency, bidirectional communication between the client and server for a smooth real-time experience.
* **Decoupled Architecture**: A robust FastAPI backend handles the heavy lifting of AI processing, while the React frontend manages the user experience.

## ⚙️ How It Works

The application is built on a client-server architecture. The frontend captures camera frames, and the backend performs the complex analysis.

1.  **Frontend (React)**: The user starts the session, granting camera access. The React app captures frames from the video feed.
2.  **WebSocket Connection**: Each captured frame is sent to the backend over a persistent WebSocket connection.
3.  **Backend (FastAPI)**:
    * The Python server receives the image data.
    * **MediaPipe**: Detects and extracts the landmark coordinates of the hand in the frame.
    * **TensorFlow Model**: The normalized landmark data is fed into a pre-trained Convolutional Neural Network (CNN) which classifies the hand sign as an ASL letter.
    * **Text Processing**: The backend logic appends the recognized letter to the current word. It uses timing and the absence of a hand to detect when a word is complete.
4.  **Real-Time Feedback**: The recognized letters and completed words are sent back to the frontend and displayed to the user instantly.
5.  **Text-to-Speech**: When the user clicks "Speak Sentence," the full text is sent to a dedicated API endpoint, which uses the Murf AI service to translate and generate an audio stream in the selected language.

## 🛠️ Technologies Used

* **Frontend**: React, `lucide-react` (icons), Tailwind CSS
* **Backend**: Python, FastAPI, Uvicorn
* **Real-Time Communication**: WebSockets
* **Machine Learning**:
    * TensorFlow & Keras (for the CNN model)
    * MediaPipe (for hand landmark detection)
    * OpenCV & NumPy (for image processing)
* **Text-to-Speech API**: Murf AI

## 🚀 Setup and Installation

To run this project locally, you will need to set up both the backend server and the frontend application.

### Prerequisites

* Python 3.8+
* Node.js and npm (or yarn)
* A Murf AI account and API key

### Backend Setup

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/hand-sign-recognition.git](https://github.com/your-username/hand-sign-recognition.git)
    cd hand-sign-recognition/backend
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing libraries like `fastapi`, `uvicorn`, `python-socketio`, `tensorflow`, `mediapipe`, `opencv-python`, `huggingface_hub`, `python-dotenv`, and `murf-ai`)*

4.  **Set Up Environment Variables**:
    Create a file named `.env` in the backend directory and add your Murf AI API key:
    ```
    MUR_API_KEY="YOUR_MURf_API_KEY_HERE"
    ```

5.  **Run the Backend Server**:
    ```bash
    uvicorn main:app --reload
    ```
    The server will be running at `http://localhost:8000`.

### Frontend Setup

1.  **Navigate to the Frontend Directory**:
    ```bash
    cd ../frontend
    ```

2.  **Install Dependencies**:
    ```bash
    npm install
    ```

3.  **Start the React Application**:
    ```bash
    npm start
    ```
    The application will open in your browser at `http://localhost:3000`.

## 📖 Usage

1.  Open `http://localhost:3000` in your web browser.
2.  Ensure the backend server is running. The status indicator in the header should say "Connected".
3.  Select your desired output language for the text-to-speech feature from the dropdown menu.
4.  Click the **Start** button to activate your webcam.
5.  Position your hand clearly in the camera frame and begin making ASL alphabet signs.
6.  The recognized letters will appear on the screen, forming words. Use the 'space' sign between words.
7.  Once you have formed a sentence, click **Stop**.
8.  Click the **Speak Sentence** button to hear the recognized text translated and spoken in your selected language.
9.  Click **Clear** to reset the text at any time.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.