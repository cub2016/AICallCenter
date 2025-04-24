# Install the assemblyai package by executing the command `pip3 install assemblyai` (macOS) or `pip install assemblyai` (Windows).
import io
import os
import time

from pyannote.audio import Pipeline
import whisper

# Import the AssemblyAI module
import assemblyai as aai
from pprint import pprint
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from pydub import AudioSegment
import numpy as np

from segment_wave_files import segment_wave_files
from transcribe_files import transcribe_segments
from transcript_analysis import transcript_analysis

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

def main():

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")

    pipeline.to(torch.device("cuda"))
    dir_list = os.listdir(location)
    for file in dir_list :
        input_file=location+file

        # assign files
        output_file = "result.wav"

        # apply pretrained pipeline
        # Pass the audio tensor and sample rate to the pipeline
        print("DIARIZING " + input_file)
        start=time.time()
        diarization = pipeline(input_file, num_speakers=2)
        print("Elapsed "+ str(time.time()-start))
            # {"waveform": audio_tensor, "sample_rate": sample_rate_tensor})
        speakers = []
        contSpeaker = ""
        dict = None
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if contSpeaker != speaker:
                if dict is not None:
                    speakers.append(dict)
                dict = {'speaker': speaker, 'start': round(turn.start, 1),
                        'end': round(turn.end, 1)}
                contSpeaker = speaker
            else:
                dict['end']= round(turn.end, 1)


        speakers = segment_wave_files(speakers, input_file)

        transcript = transcribe_segments(speakers)
        pprint(transcript)

        summary = transcript_analysis(transcript)
        pprint(summary)

import datetime

def gtWindow(number1, number2, window):
       if number1+window>number2: return True
       else: return False
j=0
def alignDiarizationToTranscript(diary, transcript):
    global j

    # cluster speakers
    test = diary.itertracks(yield_label=True)
    speakers=[]
    currSpeaker=""
    for turn, _, speaker in diary.itertracks(yield_label=True):
        if currSpeaker  != speaker:
            currSpeaker = speaker
            dict = {'speaker':currSpeaker, 'start':round(turn.start, 1),
                             'end':round(turn.end, 1)}
            speakers.append(dict)
            continue
        # just update speaker's end time
        speakers[-1]['end']=round(turn.end, 1)

    # go through transcript and align to speaker
    segments = transcript["segments"]
    segIndex = 0
    for speaker in speakers:
        speaker['transcripts']=[]
        while True:
            if speaker['end']+0.5>segments[segIndex]['start'] and speaker['end']+1.0>segments[segIndex]['end']:
                speaker['transcripts'].append(segments[segIndex]['text'])
                segIndex+=1
                print(f"segIndex = {segIndex}")
                print(f"len(segments) = {len(segments)}")
                print(segIndex >= len(segments))
                test = segIndex >= len(segments)
                print(f"test = {test}")
                if test: break
            else: break

        test = segIndex >= len(segments)
        print(f"test2 = {test}")
        if test: break
        if segIndex>= len(segments):

            break

     # print the result
    out_file = open("out_stuff_" + str(j) + "_diarization.txt", "w")
    i = 0
    for speaker in speakers:
        print(f"{speaker['speaker']} : start={speaker['start']}s stop={speaker['end']}s -->>\n {speaker['transcripts']}", file=out_file)

    out_file.close()


    #        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}", file=out_file)

    out_file = open("out_stuff_"+str(j)+"_transcript.txt", "w")
    for seg_dict in transcript["segments"]:
        print(f"start={seg_dict['start'] :.1f}s stop={seg_dict['end']:.1f}s --> {seg_dict['text']}", file=out_file)

    out_file.close()

    # out_file = open("out_stuff_"+str(j)+"_diarization.txt", "w")
    #
    # # print the result
    # i=0
    # test = diary.itertracks(yield_label=True)
    # for turn, _, speaker in diary.itertracks(yield_label=True):
    #     if i>20: break
    #     i+=1
    #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}", file=out_file)
    #
    # out_file.close()
    j+=1


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

main()
