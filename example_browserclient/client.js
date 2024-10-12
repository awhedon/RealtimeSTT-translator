let socket;
let originalTextDisplay = document.getElementById('originalTextDisplay');
let translatedTextDisplay = document.getElementById('translatedTextDisplay');
let server_available = false;
let mic_available = false;
let originalSentences = [];
let translatedSentences = [];
let pendingOriginalSentences = [];
let transcriptCount = 0;
let translationCount = 0;
let lastSpeaker = 'Doctor'; // Default to Doctor

const serverCheckInterval = 5000; // Check every 5 seconds

function connectToServer() {
    socket = new WebSocket("ws://localhost:8001");

    socket.onopen = function(event) {
        console.log("WebSocket connection established");
        server_available = true;
        start_msg();
    };

    socket.onmessage = function(event) {
        console.log("Received message:", event.data);
        let data = JSON.parse(event.data);

        if (data.type === 'realtime') {
            if (data.text.startsWith("Doctor:") || data.text.startsWith("Patient:")) {
                handleTranslation(data.text);
            } else {
                handleTranscript(data.text);
            }
        } else if (data.type === 'fullSentence') {
            if (data.text.startsWith("Doctor:") || data.text.startsWith("Patient:")) {
                handleTranslation(data.text, true);
            } else {
                handleTranscript(data.text, true);
            }
        }
    };

    socket.onclose = function(event) {
        console.log("WebSocket connection closed");
        server_available = false;
        setTimeout(connectToServer, 1000); // Try to reconnect after 1 second
    };

    socket.onerror = function(error) {
        console.error("WebSocket error:", error);
    };
}

function handleTranscript(text, isFullSentence = false) {
    pendingOriginalSentences.push({text: text, id: transcriptCount});
    transcriptCount++;
    updateDisplay();
}

function handleTranslation(text, isFullSentence = false) {
    let speaker = text.startsWith("Doctor:") ? "Doctor" : "Patient";
    lastSpeaker = speaker;
    
    if (isFullSentence) {
        translatedSentences.push(text);
    }

    // Move corresponding original sentence to the matched sentences
    if (pendingOriginalSentences.length > 0) {
        let originalSentence = pendingOriginalSentences.shift();
        originalSentences.push(`${speaker}: ${originalSentence.text}`);
    }

    translationCount++;
    updateDisplay();
}

function updateDisplay() {
    displayText(originalTextDisplay, originalSentences, pendingOriginalSentences);
    displayText(translatedTextDisplay, translatedSentences);
}

function displayText(displayDiv, sentences, pendingSentences = []) {
    let displayedText = sentences.map((sentence, index) => {
        let span = document.createElement('span');
        span.textContent = sentence + "\n";
        span.className = index % 2 === 0 ? 'yellow' : 'cyan';
        return span.outerHTML;
    }).join('');

    if (pendingSentences.length > 0) {
        let pendingText = pendingSentences.map(sentence => {
            let span = document.createElement('span');
            span.textContent = `${lastSpeaker}: ${sentence.text}\n`;
            span.className = 'pending';
            return span.outerHTML;
        }).join('');
        displayedText += pendingText;
    }

    displayDiv.innerHTML = displayedText;
    displayDiv.scrollTop = displayDiv.scrollHeight; // Auto-scroll to the bottom
}

function start_msg() {
    let message = "";
    if (!mic_available)
        message = "🎤  please allow microphone access  🎤";
    else if (!server_available)
        message = "🖥️  please start server  🖥️";
    else
        message = "👄  start speaking  👄";
    
    originalTextDisplay.innerHTML = message;
    translatedTextDisplay.innerHTML = "";
}

// Initial connection attempt
connectToServer();

// Check server availability periodically
setInterval(() => {
    if (!server_available) {
        connectToServer();
    }
}, serverCheckInterval);

// Request access to the microphone
navigator.mediaDevices.getUserMedia({ audio: true })
.then(stream => {
    let audioContext = new AudioContext();
    let source = audioContext.createMediaStreamSource(stream);
    let processor = audioContext.createScriptProcessor(1024, 1, 1);

    source.connect(processor);
    processor.connect(audioContext.destination);
    mic_available = true;
    start_msg();

    processor.onaudioprocess = function(e) {
        if (!server_available) return;

        let inputData = e.inputBuffer.getChannelData(0);
        let outputData = new Int16Array(inputData.length);

        // Convert to 16-bit PCM
        for (let i = 0; i < inputData.length; i++) {
            outputData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
        }

        // Send the 16-bit PCM data to the server
        if (socket.readyState === WebSocket.OPEN) {
            // Create a JSON string with metadata
            let metadata = JSON.stringify({ sampleRate: audioContext.sampleRate });
            // Convert metadata to a byte array
            let metadataBytes = new TextEncoder().encode(metadata);
            // Create a buffer for metadata length (4 bytes for 32-bit integer)
            let metadataLength = new ArrayBuffer(4);
            let metadataLengthView = new DataView(metadataLength);
            // Set the length of the metadata in the first 4 bytes
            metadataLengthView.setInt32(0, metadataBytes.byteLength, true); // true for little-endian
            // Combine metadata length, metadata, and audio data into a single message
            let combinedData = new Blob([metadataLength, metadataBytes, outputData.buffer]);
            socket.send(combinedData);
        }
    };
})
.catch(e => {
    console.error("Error accessing microphone:", e);
    mic_available = false;
    start_msg();
});