# Install the assemblyai package by executing the command `pip3 install assemblyai` (macOS) or `pip install assemblyai` (Windows).
import io
import os
import time 
from pyannote.audio import Pipeline

# Import the AssemblyAI module
from pprint import pprint
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from pydub import AudioSegment
import numpy as np

from segment_wave_files import segment_wave_files
from transcribe_files import transcribe_segments
from transcript_analysis import transcript_analysis
from process_audio import process_audio

#from huggingface_hub import login
#login()
hugging_face = os.environ.get("HUGGING_FACE")
pipelineDiary = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hugging_face)

pipelineDiary.to(torch.device("cuda"))

def diarize_wav_file(file_name):

    tmp_file = process_audio(file_name)

    print("DIARIZING " + file_name)
    start = time.time()
    diarization = pipelineDiary(tmp_file, num_speakers=2)
    print("Elapsed " + str(time.time() - start))
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
            dict['end'] = round(turn.end, 1)

    return speakers, tmp_file

location = os.path.join(".", "data") + os.sep
#location = os.path.join("workspace", "AICallCenter", "data") + os.sep

def get_included_files():
    files = os.listdir(location)

    return location, files

def main():

    dir_list = os.listdir(location)
    for file in dir_list :
        input_file=location+file
        #input_file='C:\\Users\\jerry\\Downloads\\SampleCallsWave\\Tech Support Help from Call Center Experts1.wav'

        # apply pretrained pipeline
        # Pass the audio tensor and sample rate to the pipeline
        speakers, tmp_file = diarize_wav_file(input_file)

        speakers = segment_wave_files(speakers, tmp_file)
        os.remove(tmp_file)
        transcript = transcribe_segments(speakers)
        print(
            "---------------------------------------------------------------------")
        pprint(transcript)
        print("---------------------------------------------------------------------")

        summary = transcript_analysis(transcript)
        pprint(summary) #.encode('utf-8').decode('utf-8'))
        print("\n\n\n\n\n\n\n")

def convertMp3ToWav(file) :
    # convert mp3 file to a wav file
    sound = AudioSegment.from_mp3(file)
    # sound.export(output_file, format="wav")

    sample_rate = sound.frame_count() / sound.duration_seconds
    print(sample_rate)
    duration = sound.duration_seconds
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    outFile = os.path.splitext(file)[0]+".wav"
    sound.export(outFile, format="wav")
    return outFile

main()
