// import React, { useState, useRef, useEffect, useCallback } from 'react';
// import { Camera, Square, Play, Pause, Volume2, XCircle } from 'lucide-react'; 

// // This function calls your backend to get the audio
// const textToSpeech = async (text, language) => {
//   try {
//     const response = await fetch('http://localhost:8000/generate-speech', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ text, language }),
//     });

//     if (!response.ok) {
//       const errorData = await response.json();
//       console.error('Error generating speech:', errorData.detail);
//       return;
//     }

//     const audioBlob = await response.blob();
//     const audioUrl = URL.createObjectURL(audioBlob);
//     const audio = new Audio(audioUrl);
//     audio.play();
    
//     audio.onended = () => {
//       URL.revokeObjectURL(audioUrl);
//     };

//   } catch (error) {
//     console.error('Failed to fetch audio:', error);
//   }
// };

// const HandSignRecognition = () => {
//   const [isActive, setIsActive] = useState(false);
//   const [currentWord, setCurrentWord] = useState('');
//   const [completedWords, setCompletedWords] = useState([]);
//   const [currentSign, setCurrentSign] = useState('');
//   const [isConnected, setIsConnected] = useState(false);
//   const [selectedLanguage, setSelectedLanguage] = useState('english');
//   const [debugInfo, setDebugInfo] = useState({ hands: 0, model: false }); 
  
//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);
//   const wsRef = useRef(null);
//   const streamRef = useRef(null);
//   const intervalRef = useRef(null); 

//   const languages = [
//     { code: 'english', name: 'English' },
//     { code: 'hindi', name: 'हिंदी (Hindi)' },
//     { code: 'spanish', name: 'Español' },
//     { code: 'french', name: 'Français' }
//   ]; 

//   const connectWebSocket = useCallback(() => {
//     if (wsRef.current) wsRef.current.close();
    
//     try {
//       wsRef.current = new WebSocket('ws://localhost:8000/ws');
      
//       wsRef.current.onopen = () => {
//         setIsConnected(true);
//         console.log('✅ WebSocket connected successfully'); 
//       };
      
//       wsRef.current.onmessage = (event) => {
//         const data = JSON.parse(event.data);
//         console.log('📨 Received data:', data); 
        
//         if (data.type === 'processing') {
//           setCurrentSign(data.current_sign || '');
//           setCurrentWord(data.current_word || ''); 
//           if (data.debug) {
//             setDebugInfo({ hands: data.debug.hands_detected, model: data.debug.model_loaded }); 
//           }
//         } else if (data.type === 'word_completed') {
//           // Add the completed word to the list
//           setCompletedWords(prev => [...prev, data.word]);
//           // Clear the current word to begin a new one
//           setCurrentWord('');
//           setCurrentSign(''); 
//           console.log(`✅ Word completed: ${data.word}`); 
//         } else if (data.type === 'error') {
//           console.error('❌ Backend error:', data.message); 
//         }
//       };
      
//       wsRef.current.onerror = (error) => {
//         console.error('❌ WebSocket error:', error);
//         setIsConnected(false); 
//       };
      
//       wsRef.current.onclose = (event) => {
//         setIsConnected(false);
//         console.log('🔌 WebSocket disconnected:', event.code, event.reason); 
//       };
//     } catch (error) {
//       console.error('❌ Failed to create WebSocket connection:', error);
//       setIsConnected(false); 
//     }
//   }, []);

//   const startCamera = async () => {
//     try {
//       streamRef.current = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
//       if (videoRef.current) {
//         videoRef.current.srcObject = streamRef.current;
//         await videoRef.current.play(); 
//       }
//       intervalRef.current = setInterval(captureAndSendFrame, 200); 
//     } catch (error) {
//       console.error('Error accessing camera:', error);
//       alert('Could not access camera. Please ensure camera permissions are granted.'); 
//     }
//   };

//   const stopCamera = () => {
//     if (streamRef.current) {
//       streamRef.current.getTracks().forEach(track => track.stop());
//       streamRef.current = null; 
//     }
//     if (intervalRef.current) {
//       clearInterval(intervalRef.current);
//       intervalRef.current = null; 
//     }
//     if (videoRef.current) {
//       videoRef.current.srcObject = null; 
//     }
//   };

//   const captureAndSendFrame = () => {
//     if (!videoRef.current || !canvasRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return; 

//     const canvas = canvasRef.current;
//     const video = videoRef.current;
//     const ctx = canvas.getContext('2d');
//     if (video.videoWidth === 0) return; 
    
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     ctx.drawImage(video, 0, 0);
//     const dataURL = canvas.toDataURL('image/jpeg', 0.8); 
    
//     wsRef.current.send(JSON.stringify({ type: 'frame', data: dataURL })); 
//   };

//   const handleToggle = useCallback(async () => {
//     if (isActive) {
//       stopCamera();
//       setIsActive(false);
//     } else {
//       if (!isConnected) connectWebSocket();
//       await new Promise(resolve => setTimeout(resolve, 1000));
//       if (wsRef.current?.readyState === WebSocket.OPEN) {
//         await startCamera();
//         setIsActive(true);
//       } else {
//         alert('Could not connect to server.');
//       }
//     }
//   }, [isActive, isConnected, connectWebSocket]); 

//   const handleLanguageChange = useCallback(async (language) => {
//     try {
//       const response = await fetch('http://localhost:8000/set_language', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ language }),
//       });
//       if (response.ok) setSelectedLanguage(language); 
//     } catch (error) {
//       console.error('Error changing language:', error); 
//     }
//   }, []);

//   const handleClear = useCallback(() => {
//     setCompletedWords([]);
//     setCurrentWord('');
//     setCurrentSign('');
//     if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
//       wsRef.current.send(JSON.stringify({ type: 'reset' }));
//     }
//   }, []); 

//   const handleSpeakSentence = useCallback(() => {
//     const fullSentence = completedWords.join(' ');
//     if (fullSentence.trim().length > 0) {
//       textToSpeech(fullSentence, selectedLanguage);
//     }
//   }, [completedWords, selectedLanguage]);

//   useEffect(() => {
//     connectWebSocket();
//     return () => {
//       stopCamera();
//       if (wsRef.current) wsRef.current.close();
//     };
//   }, [connectWebSocket]); 

//   return (
//     <div className="min-h-screen bg-gray-50 font-sans">
//        <header className="bg-white shadow-md">
//         <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col md:flex-row justify-between items-center">
//           <h1 className="text-2xl font-bold text-gray-800">Hand Sign Recognition</h1>
//           <div className="flex items-center space-x-4 mt-2 md:mt-0">
//             <select
//               value={selectedLanguage}
//               onChange={(e) => handleLanguageChange(e.target.value)}
//               className="px-3 py-1 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
//             >
//               {languages.map(lang => <option key={lang.code} value={lang.code}>{lang.name}</option>)} 
//             </select>
//             <span className={`text-sm font-medium ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
//               {isConnected ? 'Connected' : 'Disconnected'} 
//             </span>
//           </div>
//         </div>
//       </header>

//       <main className="max-w-7xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
//         <div className="bg-white p-4 rounded-lg shadow-lg">
//           <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
//             <video ref={videoRef} className="w-full h-full" playsInline muted /> 
//             <canvas ref={canvasRef} className="hidden" />
//             {!isActive && <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white font-semibold">Camera is off</div>} 
//             {isActive && currentSign && <div className="absolute top-2 right-2 bg-blue-500 text-white px-3 py-1 rounded-full text-lg font-bold">{currentSign}</div>} 
//           </div>
//           <div className="mt-4 flex flex-col md:flex-row justify-center space-y-2 md:space-y-0 md:space-x-4">
//             <button
//               onClick={handleToggle}
//               disabled={!isConnected}
//               className={`px-6 py-2 rounded-lg text-white font-semibold transform transition-transform duration-200 hover:scale-105 disabled:bg-gray-400 disabled:cursor-not-allowed
//                           ${isActive ? 'bg-red-500' : 'bg-green-500'}`}
//             >
//               {isActive ? <><Pause className="inline mr-2" size={20}/>Stop</> : <><Play className="inline mr-2" size={20}/>Start</>} 
//             </button>
//             <button
//               onClick={handleSpeakSentence}
//               disabled={completedWords.length === 0}
//               className="px-6 py-2 rounded-lg bg-blue-500 text-white font-semibold transform transition-transform duration-200 hover:scale-105 disabled:bg-gray-400 disabled:cursor-not-allowed"
//             >
//               <Volume2 className="inline mr-2" size={20}/>Speak Sentence
//             </button>
//           </div>
//         </div>

//         <div className="bg-white p-4 rounded-lg shadow-lg flex flex-col">
//           <div className="flex justify-between items-center mb-2">
//             <h2 className="text-xl font-semibold text-gray-800">Recognized Text</h2>
//             <button onClick={handleClear} className="text-sm text-gray-500 hover:text-red-500 transition-colors duration-200">
//               <XCircle className="inline mr-1" size={16}/>Clear
//             </button>
//           </div>
//           <div className="bg-gray-100 p-4 rounded-lg flex-1 overflow-y-auto">
//             <div className="text-xl font-mono text-gray-700 leading-relaxed break-words">
//               {completedWords.map((word, index) => <span key={index}>{word}{' '}</span>)}
//               <span className="text-blue-500 font-semibold">{currentWord || (isActive ? '...' : '')}</span>
//             </div>
//           </div>
//         </div>
//       </main>
//     </div>
//   );
// };

// export default HandSignRecognition;

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Play, Pause, Volume2, XCircle, Mic } from 'lucide-react';

// Hardcoded text to be "recognized" initially
const INITIAL_SENTENCE = "HELLO WORLD HOW ARE YOU";

const HandSignRecognition = () => {
  const [isActive, setIsActive] = useState(false);
  const [currentWord, setCurrentWord] = useState('');
  const [completedWords, setCompletedWords] = useState([]);
  const [currentSign, setCurrentSign] = useState('');
  
  // State for user-updatable sentence
  const [textToSimulate, setTextToSimulate] = useState(INITIAL_SENTENCE);

  // State for browser-based Text-to-Speech
  const [voices, setVoices] = useState([]);
  const [selectedVoiceURI, setSelectedVoiceURI] = useState('');

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const simulationTimeoutRef = useRef(null);

  // --- Text-to-Speech using Web Speech API ---
  const populateVoiceList = useCallback(() => {
    const availableVoices = window.speechSynthesis.getVoices();
    if (availableVoices.length > 0) {
      setVoices(availableVoices);
      // Set a default voice, preferring a local English one
      const defaultVoice = availableVoices.find(voice => voice.lang.includes('en') && voice.localService) || availableVoices[0];
      if (defaultVoice) {
        setSelectedVoiceURI(defaultVoice.voiceURI);
      }
    }
  }, []);

  useEffect(() => {
    populateVoiceList();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = populateVoiceList;
    }
  }, [populateVoiceList]);

  // This function is for client-side text-to-speech.
  const textToSpeech = useCallback((text, voiceURI) => {
    if (!text || !('speechSynthesis' in window)) {
      console.error('Speech Synthesis not supported or no text provided.');
      return;
    }
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    const selectedVoice = voices.find(voice => voice.voiceURI === voiceURI);

    if (selectedVoice) {
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
    }
    
    utterance.pitch = 1;
    utterance.rate = 0.9;
    utterance.volume = 1;
    
    window.speechSynthesis.speak(utterance);
  }, [voices]);


  // --- Camera Controls ---
  const startCamera = async () => {
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
        await videoRef.current.play();
      }
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
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };


  // --- Simulation Logic ---
  const runTextSimulation = useCallback(() => {
    setCompletedWords([]);
    setCurrentWord('');
    setCurrentSign('');

    const words = textToSimulate.toUpperCase().split(' ').filter(w => w);
    if (words.length === 0) {
        setIsActive(false);
        stopCamera();
        return;
    }

    let wordIndex = 0;

    const processWord = () => {
      if (wordIndex >= words.length) {
        setIsActive(false);
        stopCamera();
        return;
      }

      const currentWordToProcess = words[wordIndex];
      let charIndex = 0;

      const processCharacter = () => {
        if (charIndex < currentWordToProcess.length) {
          const char = currentWordToProcess[charIndex];
          setCurrentSign(char);
          setCurrentWord(prev => prev + char);
          
          charIndex++;
          simulationTimeoutRef.current = setTimeout(processCharacter, 1200); // Slightly faster simulation
        } else {
          setCompletedWords(prev => [...prev, currentWordToProcess]);
          setCurrentWord('');
          setCurrentSign('');
          
          wordIndex++;
          simulationTimeoutRef.current = setTimeout(processWord, 1800);
        }
      };
      
      processCharacter();
    };
    
    processWord();
  }, [textToSimulate]);

  const stopTextSimulation = useCallback(() => {
    clearTimeout(simulationTimeoutRef.current);
  }, []);


  // --- Event Handlers ---
  const handleToggle = useCallback(() => {
    if (isActive) {
      stopCamera();
      stopTextSimulation();
      setIsActive(false);
    } else {
      startCamera();
      runTextSimulation();
      setIsActive(true);
    }
  }, [isActive, runTextSimulation, stopTextSimulation]);

  const handleClear = useCallback(() => {
    setCompletedWords([]);
    setCurrentWord('');
    setCurrentSign('');
    if (isActive) {
      stopTextSimulation();
      stopCamera();
      setIsActive(false);
    }
  }, [isActive, stopTextSimulation]);

  const handleSpeakSentence = useCallback(() => {
    const fullSentence = completedWords.join(' ');
    if (fullSentence.trim().length > 0) {
      textToSpeech(fullSentence, selectedVoiceURI);
    }
  }, [completedWords, selectedVoiceURI, textToSpeech]);
  
  // Cleanup effect
  useEffect(() => {
    return () => {
      stopCamera();
      stopTextSimulation();
      window.speechSynthesis.cancel(); // Also cancel speech on unmount
    };
  }, [stopTextSimulation]);

  return (
    <div className="min-h-screen bg-gray-100 font-sans text-gray-800">
      <header className="bg-white shadow-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
          <h1 className="text-3xl font-bold text-gray-900">
            Hand Sign Recognition <span className="text-blue-600">Simulator</span>
          </h1>
          <div className="flex items-center space-x-3">
             <Mic className="text-gray-500"/>
             <select
               value={selectedVoiceURI}
               onChange={(e) => setSelectedVoiceURI(e.target.value)}
               className="px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
               aria-label="Select a voice for text-to-speech"
             >
               {voices.map(voice => (
                 <option key={voice.voiceURI} value={voice.voiceURI}>
                   {`${voice.name} (${voice.lang})`}
                 </option>
               ))}
                {voices.length === 0 && <option>Loading voices...</option>}
             </select>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Camera and Controls */}
        <div className="bg-white p-6 rounded-xl shadow-lg flex flex-col gap-6">
          <div className="relative w-full bg-black rounded-lg overflow-hidden aspect-video shadow-inner">
            <video ref={videoRef} className="w-full h-full object-cover" playsInline muted />
            {!isActive && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg font-semibold">
                Camera is off
              </div>
            )}
            {isActive && currentSign && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="bg-blue-600 bg-opacity-80 text-white w-32 h-32 rounded-full flex items-center justify-center text-7xl font-bold border-4 border-white shadow-lg">
                  {currentSign}
                </div>
              </div>
            )}
          </div>
          <div className="flex flex-col sm:flex-row justify-center items-center gap-4">
            <button
              onClick={handleToggle}
              className={`w-full sm:w-auto flex items-center justify-center px-8 py-3 rounded-lg text-white font-semibold transform transition-all duration-200 ease-in-out hover:scale-105 shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2
                ${isActive ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500' : 'bg-green-600 hover:bg-green-700 focus:ring-green-500'}`}
            >
              {isActive ? <><Pause className="mr-2" size={20}/>Stop Simulation</> : <><Play className="mr-2" size={20}/>Start Simulation</>}
            </button>
            <button
              onClick={handleSpeakSentence}
              disabled={completedWords.length === 0 || isActive}
              className="w-full sm:w-auto flex items-center justify-center px-8 py-3 rounded-lg bg-blue-600 text-white font-semibold transform transition-all duration-200 ease-in-out hover:scale-105 shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed disabled:scale-100"
            >
              <Volume2 className="mr-2" size={20}/>Speak Result
            </button>
          </div>
        </div>

        {/* Right Column: Text Input and Output */}
        <div className="bg-white p-6 rounded-xl shadow-lg flex flex-col gap-4">
            <div>
              <label htmlFor="text-input" className="block text-lg font-semibold text-gray-700 mb-2">
                Text to Simulate
              </label>
              <textarea
                id="text-input"
                value={textToSimulate}
                onChange={(e) => setTextToSimulate(e.target.value)}
                disabled={isActive}
                rows="3"
                className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition disabled:bg-gray-100"
                placeholder="Enter text here..."
              />
            </div>
            <div className="flex justify-between items-center">
              <h2 className="text-lg font-semibold text-gray-700">Recognized Text</h2>
              <button onClick={handleClear} className="flex items-center text-sm text-gray-500 hover:text-red-600 transition-colors duration-200 font-medium">
                <XCircle className="mr-1" size={16}/>Clear
              </button>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg flex-1 min-h-[150px] overflow-y-auto shadow-inner">
              <p className="text-2xl font-mono text-gray-900 leading-relaxed break-words">
                {completedWords.map((word, index) => <span key={index}>{word}{' '}</span>)}
                <span className="text-blue-600 font-bold border-b-2 border-blue-600 animate-pulse">
                  {currentWord}
                </span>
                {completedWords.length === 0 && currentWord === '' && !isActive && <span className="text-gray-400">Result will appear here...</span>}
              </p>
            </div>
        </div>
      </main>
    </div>
  );
};

export default HandSignRecognition;
