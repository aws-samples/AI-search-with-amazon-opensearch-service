import boto3
import json
#from IPython.display import clear_output, display, display_markdown, Markdown
import pandas as pd 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
import streamlit as st
#from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
#import torch

region = st.session_state.REGION
bedrock_runtime_client = boto3.client('bedrock-runtime',region_name=region)


# def generate_image_captions_ml():
#     model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     max_length = 16
#     num_beams = 4
#     gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def invoke_model(input):
    response = bedrock_runtime_client.invoke_model(
        body=json.dumps({
            'inputText': input
        }),
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json",
    )
    
    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")

def invoke_model_mm(text,img):
    body_ = {
            "inputText": text,
            
        }
    if(img!='none'):
        body_['inputImage']=img

    body = json.dumps(body_)
        
    modelId = 'amazon.titan-embed-image-v1'
    accept = 'application/json'
    contentType = "application/json"

    response = bedrock_runtime_client.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
    response_body = json.loads(response.get("body").read())
    #print(response_body)
    return response_body.get("embedding")

def invoke_llm_model(input,is_stream):
    if(is_stream == False):
        response = bedrock_runtime_client.invoke_model( 
            modelId= "anthropic.claude-3-sonnet-20240229-v1:0",#"anthropic.claude-3-5-sonnet-20240620-v1:0",,
            contentType = "application/json",
            accept = "application/json",
   
            body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "temperature": 0.001,
                        "top_k": 250,
                        "top_p": 1,
                        "stop_sequences": [
                            "\n\nHuman:"
                        ],
                        "messages": [
                        {
                            "role": "user",
                            "content":input
                            }
                            ]
                        }
                        
                         )
            )
        
        res = (response.get('body').read()).decode()
        
        return (json.loads(res))['content'][0]['text']
        
        # response = bedrock_runtime_client.invoke_model_with_response_stream(
        # body=json.dumps({
        #     "prompt": input,
        #     "max_tokens_to_sample": 300,
        #     "temperature": 0.5,
        #     "top_k": 250,
        #     "top_p": 1,
        #     "stop_sequences": [
        #         "\n\nHuman:"
        #     ],
        #     # "anthropic_version": "bedrock-2023-05-31"
        # }),
        # modelId="anthropic.claude-v2:1",
        # accept="application/json",
        # contentType="application/json",
        # )
        # stream = response.get('body')
        
        # return stream
        
    # else:
    #     response = bedrock_runtime_client.invoke_model_with_response_stream( 
    #         modelId= "anthropic.claude-3-sonnet-20240229-v1:0",
    #         contentType = "application/json",
    #         accept = "application/json",
   
    #         body = json.dumps({
    #                     "anthropic_version": "bedrock-2023-05-31",
    #                     "max_tokens": 1024,
    #                     "temperature": 0.0001,
    #                     "top_k": 150,
    #                     "top_p": 0.7,
    #                     "stop_sequences": [
    #                         "\n\nHuman:"
    #                     ],
    #                     "messages": [
    #                     {
    #                         "role": "user",
    #                         "content":input
    #                         }
    #                         ]
    #                     }
                        
    #                      )
    #         )
        
    #     stream = response.get('body')
        
    #     return stream
        
def read_from_table(file,question):
    print("started table analysis:")
    print("-----------------------")
    print("\n\n")
    print("Table name: "+file)
    print("-----------------------")
    print("\n\n")
    bedrock_params = {
    "max_tokens":2048,
    "temperature":0.0001,
    "top_k":150,
    "top_p":0.7,
    "stop_sequences":["\\n\\nHuman:"]
    }
    
    model = BedrockChat(
    client=bedrock_runtime_client,
    model_id='anthropic.claude-3-sonnet-20240229-v1:0',
    model_kwargs=bedrock_params,
    streaming=False
    )
    if(str(type(file))=="<class 'str'>"):
        df = pd.read_csv(file,skipinitialspace = True, on_bad_lines='skip',delimiter = "`")
    else:
        df = file
    #df.fillna(method='pad', inplace=True)
    agent = create_pandas_dataframe_agent(
             model, 
             df, 
             verbose=True,
             agent_executor_kwargs={'handle_parsing_errors':True,
                                    'return_only_outputs':True}
             )
    agent_res = agent.invoke(question)['output']
    return agent_res
    
def generate_image_captions_llm(base64_string,question):
    
    # ant_client = Anthropic()
    # MODEL_NAME = "claude-3-opus-20240229"
        
    # message_list = [
    # {
    #     "role": 'user',
    #     "content": [
    #         {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_string}},
    #         {"type": "text", "text": "What is in the image ?"}
    #     ]
    # }
    # ]

    # response = ant_client.messages.create(
    # model=MODEL_NAME,
    # max_tokens=2048,
    # messages=message_list
    # )
    response = bedrock_runtime_client.invoke_model( 
            modelId= "anthropic.claude-3-sonnet-20240229-v1:0",
            contentType = "application/json",
            accept = "application/json",
   
            body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "messages": [
                        {
                            "role": "user",
                            "content": [
                            {
                                "type": "image",
                                "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_string
                                }
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                            ]
                        }
                        ]
                         }))
    #print(response)
    response_body = json.loads(response.get("body").read())['content'][0]['text']

    #print(response_body)
    
    return response_body