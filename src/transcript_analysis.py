from encodings.utf_8 import encode
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import transformers
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

import torch

def use_huggingface(input):
    # Define the model name
    # model_name = "cnicu/t5-small-booksum"
    # model_name = "sshleifer/distilbart-cnn-12-6"
    #model_name = "facebook/bart-large-cnn"
    model_name = "huggyllama/llama-7b"
    #model_name = "google/pegasus-large"

#    summarizer = pipeline(task="summarization", model=model_name, device_map="cuda")
    summarizer = pipeline(task="summarization", model=model_name,
                          torch_dtype=torch.float16)


    max_length = 1024
    #input = "summarize the following transcript:"+input
    # length = len(input)
    # if length < 1024:
    #     max_length = length
    # else:
    #     input = input[0:1023]
    input = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """
    # # Pass the long text to the model to summarize it with a maximum length of 50 tokens
    outputs = summarizer(input, max_length=max_length, do_sample=False)
    # , min_length = 25, max_length=500
    # Access and print the summarized text in the outputs variable
    return outputs[0]['summary_text']


from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

def use_huggingface2(input):
    from transformers import pipeline
    # Define the model name
    # model_name = "cnicu/t5-small-booksum"
    # model_name = "sshleifer/distilbart-cnn-12-6"
    #model_name = "facebook/bart-large-cnn"
    #model_name = "huggyllama/llama-7b"
    model_name = "raaec/Meta-Llama-3.1-8B-Instruct-Summarizer"
    #model_name = "google/pegasus-large"
    #model_name = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # max_length=1000,
        max_new_tokens=800,
        eos_token_id=tokenizer.eos_token_id
    )

    # Wrap in LangChain-compatible LLM
#     text ="""The Trump administration is taking a major step to super-charge deportation efforts by deputizing law enforcement across the country to carry out immigration enforcement, a Department of Homeland Security memo exclusively reviewed by The Daily Wire reveals.
# The memo lays out that DHS has signed 562 agreements with state and local law enforcement groups, taking advantage of a federal law that allows the agencies to enforce certain aspects of immigration law. While these sorts of agreements existed in the previous administration, the number of agreements signed since Trump took office is already a 316% increase from what was signed during four years of the Biden administration, the memo says.
# The agreements could be one of the biggest keys to boosting deportation numbers, which members of the administration have reportedly grown frustrated with. With an estimated 18.6 million illegal aliens currently present in the United States and a judiciary that has continuously stonewalled the Trump administration’s attempts to deport illegal aliens, the federal government may otherwise struggle to remove a significant share of the illegal population without support from law enforcement partners.
# Under the agreement, state and local law enforcement agencies are able to conduct immigration enforcement operations with ICE oversight, use federal immigration databases to identify the status of arrestees, and initiate ICE detainers to hold illegal aliens until ICE can gain custody of them, providing what could be a massive force multiplier in the Trump administration’s effort to deport at least one million illegal aliens per year.
# The administration says its collaboration with state and local law enforcement is yielding results.
# “Since January 20, our partnerships with state and local law enforcement have led to over 1,100 arrests of dangerous criminal illegal aliens,” the Homeland Security memo says. “ICE has also trained over 5,000 officers across the country to help track down and arrest criminal illegal aliens in their communities.”
# One far-left organization backed by George Soros’s Open Society Foundations complained that these agreements “are designed to extend the reach of the Trump deportation machine.” Now, they’re proving to be an effective force multiplier for the Trump administration as it seeks to enforce federal immigration law.
# One of these agreements resulted in Operation Tidal Wave, a first-of-its-kind joint immigration enforcement effort from Homeland Security and Florida."""

    template = """
    You are a helpful assistant that summarizes conversations.

    Summarize the following dialog. Identify the key points and exchanges between speakers.
    Use bullet points to describe important statements or shifts in topic.
    Preserve who said what when it's important.

    ```{text}```

    BULLET POINT SUMMARY:
    """

    llm = HuggingFacePipeline(
        pipeline=text_gen_pipeline,
        model_kwargs={
            "temperature": 0.3,  # Lower = more focused and factual
            "top_k": 50,  # Limits selection to top 50 tokens by probability
            "top_p": 0.85,  # Cumulative probability sampling
            "repetition_penalty": 1.2,  # Penalizes repeated phrases
            "max_new_tokens": 300  # Ensures full, rich output
        }
    )

    prompt = PromptTemplate(template=template, input_variables=["text"])
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    chain = prompt | llm | StrOutputParser()

    out = chain.invoke(input)
    # out = llm_chain.run(text)

 #   quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
#    model = AutoModelForCausalLM.from_pretrained(
  #      model_name,
   #     torch_dtype=torch.bfloat16,
    #    device_map="auto",
     #   quantization_config=quantization_config
#    )

#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    input_ids = tokenizer(f"""You are an helpful assistant.  Analyze the following transcript and 
#        give a summary of the conversation.  Also give the sentiment of each speaker 
#        as the conversation progresses. Also replace Speaker_0x with name or title 
#        if it can be derived from conversation. Response should only contain ascii 
#        characters. {input}""", return_tensors="pt").to("cuda")

 #   output = model.generate(**input_ids, cache_implementation="static")
 #   out = tokenizer.decode(output[0], skip_special_tokens=True)
    print(out)
    return out


def transcript_analysis(transcript):
    results = []
    #print(os.environ['OPENAI_API_KEY'])

    input=""
    for speaker in transcript:
        input += speaker + "\n"

    start = time.time()
    # response = use_huggingface2(input)
    response = use_openai(input)
    #response = use_bigbird_pegasus_large_arxiv(input)
    stop = time.time()
    elapsed=stop-start
    print("transcript analysis consumed "+str(elapsed))

    return response
