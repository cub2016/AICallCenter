from encodings.utf_8 import encode

from openai import OpenAI
import os
from pprint import pprint
from transformers import pipeline
import torch

def use_openai(input):
    client = OpenAI()
    response = client.responses.create(
        model="o3-mini-2025-01-31",
        # model="gpt-4.1",
        input=f"""You are an helpful assistant.  Analyze the following transcript and 
        give a summary of the conversation.  Also give the sentiment of each speaker 
        as the conversation progresses. Response should only contain ascii characters. {input}"""
    )
    return response.output_text

def use_huggingface(input):
    # Define the model name
    # model_name = "cnicu/t5-small-booksum"
    model_name = "sshleifer/distilbart-cnn-12-6"
    #model_name = "facebook/bart-large-cnn"

    summarizer = pipeline(task="summarization", model=model_name)
    # summarizer.to(torch.device("cuda"))

    # Pass the long text to the model to summarize it with a maximum length of 50 tokens
    outputs = summarizer("summarize the following transcript:"+input, max_length=1024, do_sample=False)
    # , min_length = 25, max_length=500
    # Access and print the summarized text in the outputs variable
    return outputs[0]['summary_text']

def transcript_analysis(transcript):
    results = []
    #print(os.environ['OPENAI_API_KEY'])

    input=""
    for speaker in transcript:
        input += speaker
        input += "\n"

    response = use_huggingface(input)
    # use_openai(input)

    # with open('filename', 'w', encoding='utf-8') as f:
    # pprint(response.output_text.decode('UTF-8'))
    return response
