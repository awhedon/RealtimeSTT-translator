if __name__ == '__main__':
    print("Starting server, please wait...")

    from RealtimeSTT import AudioToTextRecorder
    import asyncio
    import websockets
    import threading
    import numpy as np
    from scipy.signal import resample
    import json
    import re
    from ast import literal_eval
    from openai import OpenAI
    import os

    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    recorder = None
    recorder_ready = threading.Event()
    client_websocket = None

    async def send_to_client(message):
        if client_websocket:
            await client_websocket.send(message)

    def translate_text(text):
        prompt = f"""If the Text to Translate below is in English, translate it to Spanish and output "Doctor: {{translated text}}". If the Text to Translate below is in Spanish, translate it to English and output "Patient: {{translated text}}".

Text to Translate:
{text}

DO NOT OUTPUT ANYTHING OTHER THAN "Doctor: {{translated text}}" or "Patient: {{translated text}}".
"""
        output = generate_response__simple(prompt, stream=False)
        return output

    def extract_first_valid_json(text: str):
        # Regular expression patterns for potential JSON structures
        object_pattern = r'\{.*?\}'
        array_pattern = r'\[.*?\]'

        # Combine the patterns
        combined_pattern = f'({object_pattern}|{array_pattern})'

        # Search for potential JSON structures
        matches = re.finditer(combined_pattern, text, re.DOTALL)

        # Try parsing each match as JSON
        for match in matches:
            try:
                # Attempt to parse the matched string as JSON
                matches = match.group()
                json_data = json.loads(matches)

                # If successful, return the JSON data
                return json_data
            except json.JSONDecodeError as e:
                print('JSONDecodeError:', e, 'matches:', matches)
                # try again with different parse function in case the string is encoded in Python dict format
                try:
                    matches = match.group()
                    json_data = literal_eval(matches)
                    return json_data
                except ValueError:
                    # If parsing fails, continue to the next match
                    continue

        # No valid JSON found
        return {}
    
    def generate_response__simple(prompt: str, variable_dict = {}, stream: bool = True, json_mode: bool = False):
        print('generating...')
        for key in variable_dict:
            prompt = prompt.replace(key, variable_dict[key])
        if stream:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                stream=True
            )

            def generate():
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return generate()
        else:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                stream=False
            )

            output = response.choices[0].message.content
            if json_mode:
                output = extract_first_valid_json(output)
            return output

    def text_detected(text):
        asyncio.new_event_loop().run_until_complete(
            send_to_client(
                json.dumps({
                    'type': 'realtime',
                    'text': text
                })
            )
        )
        print(f"Original: {text}", flush=True)
        translated_text = translate_text(text)
        asyncio.new_event_loop().run_until_complete(
            send_to_client(
                json.dumps({
                    'type': 'realtime',
                    'text': translated_text
                })
            )
        )
        print(f"Translated: {translated_text}", flush=True)

    recorder_config = {
        'spinner': False,
        'use_microphone': False,
        # 'model': 'large-v2',
        'model': 'tiny.en',
        'language': 'en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.7,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_stabilized': text_detected,
    }

    def recorder_thread():
        global recorder
        print("Initializing RealtimeSTT...")
        recorder = AudioToTextRecorder(**recorder_config)
        print("RealtimeSTT initialized")
        recorder_ready.set()
        while True:
            full_sentence = recorder.text()
            asyncio.new_event_loop().run_until_complete(
                send_to_client(
                    json.dumps({
                        'type': 'fullSentence',
                        'text': full_sentence
                    })
                )
            )
            print(f"Full sentence: {full_sentence}")
            translated_sentence = translate_text(full_sentence)
            asyncio.new_event_loop().run_until_complete(
                send_to_client(
                    json.dumps({
                        'type': 'fullSentence',
                        'text': translated_sentence
                    })
                )
            )
            print(f"Translated full sentence: {translated_sentence}")

    def decode_and_resample(
            audio_data,
            original_sample_rate,
            target_sample_rate):

        # Decode 16-bit PCM data to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate the number of samples after resampling
        num_original_samples = len(audio_np)
        num_target_samples = int(num_original_samples * target_sample_rate /
                                 original_sample_rate)

        # Resample the audio
        resampled_audio = resample(audio_np, num_target_samples)

        return resampled_audio.astype(np.int16).tobytes()

    async def echo(websocket, path):
        print("Client connected")
        global client_websocket
        client_websocket = websocket
        async for message in websocket:

            if not recorder_ready.is_set():
                print("Recorder not ready")
                continue

            metadata_length = int.from_bytes(message[:4], byteorder='little')
            metadata_json = message[4:4+metadata_length].decode('utf-8')
            metadata = json.loads(metadata_json)
            sample_rate = metadata['sampleRate']
            chunk = message[4+metadata_length:]
            resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
            recorder.feed_audio(resampled_chunk)

    # start_server = websockets.serve(echo, "0.0.0.0", 9001)
    start_server = websockets.serve(echo, "localhost", 8001)

    recorder_thread = threading.Thread(target=recorder_thread)
    recorder_thread.start()
    recorder_ready.wait()

    print("Server started. Press Ctrl+C to stop the server.")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
