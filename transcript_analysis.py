from openai import OpenAI
import os


def transcript_analysis(transcript):
    results = []
    #print(os.environ['OPENAI_API_KEY'])

    client = OpenAI()
    input=""
    for speaker in transcript:
        input += speaker
        input += "\n"

    response = client.responses.create(
        model="o3-mini-2025-01-31",
        # model="gpt-4.1",
        input=f"""You are an helpful assistant.  Analyze teh following transcript and 
        give a summary of the conversation.  Also give the sentiment of each speaker 
        as the conversation progresses. {input}"""
    )

    print(response.output_text)
    return response.output_text
