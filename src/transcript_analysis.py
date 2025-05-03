from encodings.utf_8 import encode

from openai import OpenAI
import time
from transformers import pipeline

def use_openai(input):
    client = OpenAI()
    response = client.responses.create(
        model="o3-mini-2025-01-31",
        # model="gpt-4.1",
        input=f"""You are an helpful assistant.  Analyze the following transcript and 
        give a summary of the conversation.  Also give the sentiment of each speaker 
        as the conversation progresses. Also replace Speaker_0x with name or title 
        if it can be derived from conversation. Response should only contain ascii 
        characters. {input}"""
    )
    return response.output_text

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
def use_bigbird_pegasus_large_arxiv(input):

    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

    # by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")

    # decoder attention type can't be changed & will be "original_full"
    # you can change `attention_type` (encoder only) to full attention like this:
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv", attention_type="original_full")

    # you can change `block_size` & `num_random_blocks` like this:
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv", block_size=16, num_random_blocks=2)

    input=f"""You are an helpful assistant.  Analyze the following transcript and 
        give a summary of the conversation.  Also give the sentiment of each speaker 
        as the conversation progresses. Also replace Speaker_0x with name or title 
        if it can be derived from conversation. Response should only contain ascii 
        characters. {input}"""
    inputs = tokenizer(input, return_tensors='pt')
    prediction = model.generate(**inputs)
    prediction = tokenizer.batch_decode(prediction)


def use_huggingface(input):
    # Define the model name
    # model_name = "cnicu/t5-small-booksum"
    # model_name = "sshleifer/distilbart-cnn-12-6"
    #model_name = "facebook/bart-large-cnn"
    model_name = "google/pegasus-large"

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

    start = time.time()
    # response = use_huggingface(input)
 #   response = use_openai(input)
    response = use_bigbird_pegasus_large_arxiv(input)
    stop = time.time()
    elapsed=stop-start
    print("transcript analysis consumed "+str(elapsed))


    # with open('filename', 'w', encoding='utf-8') as f:
    # pprint(response.output_text.decode('UTF-8'))
    return response
