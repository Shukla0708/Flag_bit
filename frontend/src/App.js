import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Square, Play, Pause, Volume2, XCircle } from 'lucide-react'; 

// This function calls your backend to get the audio
const textToSpeech = async (text, language) => {
  try {
    const response = await fetch('http://localhost:8000/generate-speech', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, language }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Error generating speech:', errorData.detail);
      return;
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
    
    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
    };

  } catch (error) {
    console.error('Failed to fetch audio:', error);
  }
};

const HandSignRecognition = () => {
  const [isActive, setIsActive] = useState(false);
  const [currentWord, setCurrentWord] = useState('');
  const [completedWords, setCompletedWords] = useState([]);
  const [currentSign, setCurrentSign] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [debugInfo, setDebugInfo] = useState({ hands: 0, model: false }); 
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null); 

  const languages = [
    { code: 'english', name: 'English' },
    { code: 'hindi', name: 'हिंदी (Hindi)' },
    { code: 'spanish', name: 'Español' },
    { code: 'french', name: 'Français' }
  ]; 

  const connectWebSocket = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        console.log('✅ WebSocket connected successfully'); 
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('📨 Received data:', data); 
        
        if (data.type === 'processing') {
          setCurrentSign(data.current_sign || '');
          setCurrentWord(data.current_word || ''); 
          if (data.debug) {
            setDebugInfo({ hands: data.debug.hands_detected, model: data.debug.model_loaded }); 
          }
        } else if (data.type === 'word_completed') {
          // Add the completed word to the list
          setCompletedWords(prev => [...prev, data.word]);
          // Clear the current word to begin a new one
          setCurrentWord('');
          setCurrentSign(''); 
          console.log(`✅ Word completed: ${data.word}`); 
        } else if (data.type === 'error') {
          console.error('❌ Backend error:', data.message); 
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setIsConnected(false); 
      };
      
      wsRef.current.onclose = (event) => {
        setIsConnected(false);
        console.log('🔌 WebSocket disconnected:', event.code, event.reason); 
      };
    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error);
      setIsConnected(false); 
    }
  }, []);

  const startCamera = async () => {
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
        await videoRef.current.play(); 
      }
      intervalRef.current = setInterval(captureAndSendFrame, 200); 
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera. Please ensure camera permissions are granted.'); 
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null; 
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null; 
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null; 
    }
  };

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return; 

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    if (video.videoWidth === 0) return; 
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg', 0.8); 
    
    wsRef.current.send(JSON.stringify({ type: 'frame', data: dataURL })); 
  };

  const handleToggle = useCallback(async () => {
    if (isActive) {
      stopCamera();
      setIsActive(false);
    } else {
      if (!isConnected) connectWebSocket();
      await new Promise(resolve => setTimeout(resolve, 1000));
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        await startCamera();
        setIsActive(true);
      } else {
        alert('Could not connect to server.');
      }
    }
  }, [isActive, isConnected, connectWebSocket]); 

  const handleLanguageChange = useCallback(async (language) => {
    try {
      const response = await fetch('http://localhost:8000/set_language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language }),
      });
      if (response.ok) setSelectedLanguage(language); 
    } catch (error) {
      console.error('Error changing language:', error); 
    }
  }, []);

  const handleClear = useCallback(() => {
    setCompletedWords([]);
    setCurrentWord('');
    setCurrentSign('');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    }
  }, []); 

  const handleSpeakSentence = useCallback(() => {
    const fullSentence = completedWords.join(' ');
    if (fullSentence.trim().length > 0) {
      textToSpeech(fullSentence, selectedLanguage);
    }
  }, [completedWords, selectedLanguage]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      stopCamera();
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWebSocket]); 

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
       <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col md:flex-row justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">Hand Sign Recognition</h1>
          <div className="flex items-center space-x-4 mt-2 md:mt-0">
            <select
              value={selectedLanguage}
              onChange={(e) => handleLanguageChange(e.target.value)}
              className="px-3 py-1 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {languages.map(lang => <option key={lang.code} value={lang.code}>{lang.name}</option>)} 
            </select>
            <span className={`text-sm font-medium ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
              {isConnected ? 'Connected' : 'Disconnected'} 
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
            <video ref={videoRef} className="w-full h-full" playsInline muted /> 
            <canvas ref={canvasRef} className="hidden" />
            {!isActive && <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white font-semibold">Camera is off</div>} 
            {isActive && currentSign && <div className="absolute top-2 right-2 bg-blue-500 text-white px-3 py-1 rounded-full text-lg font-bold">{currentSign}</div>} 
          </div>
          <div className="mt-4 flex flex-col md:flex-row justify-center space-y-2 md:space-y-0 md:space-x-4">
            <button
              onClick={handleToggle}
              disabled={!isConnected}
              className={`px-6 py-2 rounded-lg text-white font-semibold transform transition-transform duration-200 hover:scale-105 disabled:bg-gray-400 disabled:cursor-not-allowed
                          ${isActive ? 'bg-red-500' : 'bg-green-500'}`}
            >
              {isActive ? <><Pause className="inline mr-2" size={20}/>Stop</> : <><Play className="inline mr-2" size={20}/>Start</>} 
            </button>
            <button
              onClick={handleSpeakSentence}
              disabled={completedWords.length === 0}
              className="px-6 py-2 rounded-lg bg-blue-500 text-white font-semibold transform transition-transform duration-200 hover:scale-105 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              <Volume2 className="inline mr-2" size={20}/>Speak Sentence
            </button>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-lg flex flex-col">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-xl font-semibold text-gray-800">Recognized Text</h2>
            <button onClick={handleClear} className="text-sm text-gray-500 hover:text-red-500 transition-colors duration-200">
              <XCircle className="inline mr-1" size={16}/>Clear
            </button>
          </div>
          <div className="bg-gray-100 p-4 rounded-lg flex-1 overflow-y-auto">
            <div className="text-xl font-mono text-gray-700 leading-relaxed break-words">
              {completedWords.map((word, index) => <span key={index}>{word}{' '}</span>)}
              <span className="text-blue-500 font-semibold">{currentWord || (isActive ? '...' : '')}</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default HandSignRecognition;
