from google.cloud import speech

def Listen(audio_file):
    client = speech.SpeechClient()
    
    # Read the audio file content
    with open(audio_file, 'rb') as f:
        audio = speech.RecognitionAudio(content=f.read())
    
    # Transcribe the speech
    response = client.recognize(config=speech.RecognitionConfig(language_code="en-US"), audio=audio)
    
    # Return the transcribed text
    return response.results[0].alternatives[0].transcript
