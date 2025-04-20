# Install the assemblyai package by executing the command `pip3 install assemblyai` (macOS) or `pip install assemblyai` (Windows).

# Import the AssemblyAI module
import assemblyai as aai
from pprint import pprint

# Your API token is already set here
# AssemblyAI key
aai.settings.api_key = "37a4985f47ea4d3c834b4e2c3abdbb15"

location="C:\\Users\\jerry\\Downloads\\SampleCalls\\"
files=[
    'RexSchool.mp3',
    'Tech Support Help from Call Center Experts1.mp3',
    'Tech Support Help from Call Center Experts2.mp3',
    'Tech Support Help from Call Center Experts3.mp3',
    'Warid_help_line.mp3',
    'angryCustomer1.mp3',
    'angryCustomer2.mp3',
    'callCenterOrder.mp3',
    'misshandledCall.mp3',
]

# Create a transcriber object.
transcriber = aai.Transcriber()
out_path='C:\\Users\\jerry\\Downloads\\transcripts\\'
# config = aai.TranscriptionConfig(filter_profanity=False,
#                                  sentiment_analysis=True,
#                                  summarization=True,
#                                  speakers_expected=2)
config = aai.TranscriptionConfig(speakers_expected=2,
                                 filter_profanity=False,
                                  sentiment_analysis=True,
                                  summarization=True,
                                 speaker_labels=True)

for file in files:
    input_file=location+file
    # Transcribe the audio with the configuration
    transcript = transcriber.transcribe(input_file, config)

    out_file= out_path + file[:file.index('.')] + '.txt'
    # Print the transcript with speaker labels
    with open(out_file, 'w') as f:
        for utterance in transcript.utterances:
            print(f"Speaker {utterance.speaker}: {utterance.text} ->-> {utterance.confidence}", file=f)
            print("--------------------------------------------", file=f)

    prompt = "Provide a brief summary of the transcript."
    result = transcript.lemur.task(
        prompt, final_model=aai.LemurModel.claude3_5_sonnet
    )

    print(result.response)

# Alternatively, if you have a URL to an audio file, you can transcribe it with the following code.
# Uncomment the line below and replace the URL with the link to your audio file.
# transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/espn-bears.m4a")

# After the transcription is complete, the text is printed out to the console.
# print(transcript.text)
#pprint(transcript.text)