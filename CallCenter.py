# Install the assemblyai package by executing the command `pip3 install assemblyai` (macOS) or `pip install assemblyai` (Windows).
import io
import os
from pyannote.audio import Pipeline

# Import the AssemblyAI module
import assemblyai as aai
from pprint import pprint
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from pydub import AudioSegment
import numpy as np

# Your API token is already set here
# AssemblyAI key
aai.settings.api_key = "37a4985f47ea4d3c834b4e2c3abdbb15"

location="C:\\Users\\jerry\\Downloads\\SampleCallsWave\\"

# Create a transcriber object.
transcriber = aai.Transcriber()
out_path='C:\\Users\\jerry\\Downloads\\transcripts\\'
# config = aai.TranscriptionConfig(filter_profanity=False,
#                                  sentiment_analysis=True,
#                                  summarization=True,
#                                  speakers_expected=2)
# config = aai.TranscriptionConfig(speakers_expected=2,
#                                  filter_profanity=False,
#                                   sentiment_analysis=True,
#                                   summarization=True,
#                                  speaker_labels=True)
dir_list = os.listdir(location)
for file in dir_list :
    input_file=location+file

    # assign files
    output_file = "result.wav"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")

    pipeline.to(torch.device("cuda"))

    # apply pretrained pipeline
    # Pass the audio tensor and sample rate to the pipeline
    diarization = pipeline(input_file)
        # {"waveform": audio_tensor, "sample_rate": sample_rate_tensor})

    id_convo = []
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        whisper.
    # Transcribe the audio with the configuration
    # transcript = transcriber.transcribe(input_file, config)
    #
    # out_file= out_path + file[:file.index('.')] + '.txt'
    # # Print the transcript with speaker labels
    # transcript_array = []
    # with open(out_file, 'w') as f:
    #     for utterance in transcript.utterances:
    #         transcript_array.append(f"Speaker {utterance.speaker}: {utterance.text}")
    #         print(f"Speaker {utterance.speaker}: {utterance.text} ->-> {utterance.confidence}", file=f)
    #         print("--------------------------------------------", file=f)
    #
    # pprint(transcript_array)
    # # prompt = "Provide a brief summary of the transcript."
    # # result = transcript.lemur.task(
    # #     prompt, final_model=aai.LemurModel.claude3_5_sonnet
    # # ))
    # #
    # # print(result.response)

# Alternatively, if you have a URL to an audio file, you can transcribe it with the following code.
# Uncomment the line below and replace the URL with the link to your audio file.
# transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/espn-bears.m4a")

# After the transcription is complete, the text is printed out to the console.
# print(transcript.text)
#pprint(transcript.text)


def convertMp3ToWav(file) :
    # convert mp3 file to wav file
    sound = AudioSegment.from_mp3(file)
    # sound.export(output_file, format="wav")

    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     #model="openai/whisper-large-v3",
    #     model="openai/whisper-large-v3-turbo",
    #     # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    #     torch_dtype=torch.float16,
    #     device="cuda:0",  # or mps for Mac devices
    #     model_kwargs={
    #         "attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {
    #         "attn_implementation": "sdpa"},
    # )
    #
    # outputs = pipe(
    #     input_file,
    #     chunk_length_s=30,
    #     batch_size=24,
    #     return_timestamps=True,
    # )
    #
    # print(outputs)

    sample_rate = sound.frame_count() / sound.duration_seconds
    print(sample_rate)
    duration = sound.duration_seconds
    num_channels = sound.channels
    num_frames = int(sample_rate * duration)
    fp_samples = np.array(sound.get_array_of_samples()).astype(np.float32)
    chan_array = np.array([num_channels] * num_frames).astype(np.float32)

    audio_array = np.stack((fp_samples, chan_array))
    # audio_array = [sound.get_array_of_samples().astype(np.float32), [num_channels]*num_frames]
    # audio_array = np.array(audio_array)

    audio_array2 = np.random.randn(num_frames, num_channels)
    # Convert numpy array to torch tensor
    audio_tensor = torch.from_numpy(audio_array)  # .T
    # audio_tensor = audio_tensor.T
    # Create a sample rate tensor
    sample_rate_tensor = torch.tensor(float(sample_rate))

# pipeline.to(torch.device("cuda"))
