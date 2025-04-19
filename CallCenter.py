# Install the assemblyai package by executing the command `pip3 install assemblyai` (macOS) or `pip install assemblyai` (Windows).

# Import the AssemblyAI module
import assemblyai as aai
from pprint import pprint

# Your API token is already set here
aai.settings.api_key = "37a4985f47ea4d3c834b4e2c3abdbb15"

# Create a transcriber object.
transcriber = aai.Transcriber()
#input_file="C:\\Users\\jerry\\Downloads\\callCenterSlow.mp4"
input_file="C:\\Users\\jerry\\Downloads\\callCenterIrateCust.mp4"
# If you have a local audio file, you can transcribe it using the code below.
# Make sure to replace the filename with the path to your local audio file.
#transcript = transcriber.transcribe(input_file)

config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=3) # Example with 2 expected speakers

# Transcribe the audio with the configuration
transcript = transcriber.transcribe(input_file, config)

# Print the transcript with speaker labels
for utterance in transcript.utterances:
#    pprint(f"Speaker {utterance.speaker}: {utterance.text} ->-> {utterance.confidence}")
    print(f"Speaker {utterance.speaker}: {utterance.text} ->-> {utterance.confidence}")
    print("--------------------------------------------")

# Alternatively, if you have a URL to an audio file, you can transcribe it with the following code.
# Uncomment the line below and replace the URL with the link to your audio file.
# transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/espn-bears.m4a")

# After the transcription is complete, the text is printed out to the console.
# print(transcript.text)
#pprint(transcript.text)