import streamlit as st
import uuid
import os
import boto3
import requests
import lambda_api
from boto3 import Session
import botocore.session
import json
import random
import string
from PIL import Image 
import urllib.request 
#from langchain.callbacks.base import BaseCallbackHandler


USER_ICON = "/home/ec2-user/AI-search-with-amazon-opensearch-service/opensearch-hybridsearch/webapp/images/user.png"
AI_ICON = "/home/ec2-user/AI-search-with-amazon-opensearch-service/opensearch-hybridsearch/webapp/images/opensearch-twitter-card.png"
REGENERATE_ICON = "/home/ec2-user/AI-search-with-amazon-opensearch-service/opensearch-hybridsearch/webapp/images/regenerate.png"
s3_bucket_ = "pdf-repo-uploads"
            #"pdf-repo-uploads"

# Check if the user ID is already stored in the session state
if 'user_id' in st.session_state:
    user_id = st.session_state['user_id']
    print(f"User ID: {user_id}")

# If the user ID is not yet stored in the session state, generate a random UUID
else:
    user_id = str(uuid.uuid4())
    st.session_state['user_id'] = user_id


if 'session_id' not in st.session_state:
    st.session_state['session_id'] = ""
    
if "chats" not in st.session_state:
    st.session_state.chats = [
        {
            'id': 0,
            'question': '',
            'answer': ''
        }
    ]

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "input_text" not in st.session_state:
    st.session_state.input_text=""

if "input_searchType" not in st.session_state:
    st.session_state.input_searchType = "Keyword Search"

# if "input_temperature" not in st.session_state:
#     st.session_state.input_temperature = "0.001"

# if "input_topK" not in st.session_state:
#     st.session_state.input_topK = 200

# if "input_topP" not in st.session_state:
#     st.session_state.input_topP = 0.95

# if "input_maxTokens" not in st.session_state:
#     st.session_state.input_maxTokens = 1024


def handle_input():
    inputs = {}
    for key in st.session_state:
        if key.startswith('input_'):
            inputs[key.removeprefix('input_')] = st.session_state[key]
    st.session_state.inputs_ = inputs
    
    #st.write(inputs) 
    question_with_id = {
        'question': inputs["text"],
        'id': len(st.session_state.questions)
    }
    st.session_state.questions = []
    st.session_state.questions.append(question_with_id)
    
    st.session_state.answers = []
    st.session_state.answers.append({
        'answer': lambda_api.call(json.dumps(inputs), st.session_state['session_id']),
        'search_type':inputs['searchType'],
        'id': len(st.session_state.questions)
    })
    #st.session_state.input_text=""
    #st.session_state.input_searchType=st.session_state.input_searchType

def write_top_bar():
    col1, col2, col3, col4 = st.columns([1.75,10,2,5])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        #st.markdown("")
        input = st.text_input( "Ask here",label_visibility = "collapsed",key="input_text")
    
    with col3:
        #hidden = st.button("RUN",disabled=True,key = "hidden")
        play = st.button("GO",on_click=handle_input,key = "play")
    with col4:
        clear = st.button("Clear Results")
    return clear

clear = write_top_bar()

if clear:
    st.session_state.questions = []
    st.session_state.answers = []
    #st.session_state.input_text=""
    # st.session_state.input_searchType="Conversational Search (RAG)"
    # st.session_state.input_temperature = "0.001"
    # st.session_state.input_topK = 200
    # st.session_state.input_topP = 0.95
    # st.session_state.input_maxTokens = 1024






    
search_type = st.selectbox('Select the Search type',
    ('Keyword Search',
    'Vector Search', 
    'Hybrid Search'
    ),
   
    key = 'input_searchType',
    help = "Select the type of retriever\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers)"
    )

with st.sidebar:
    st.header('Fine-tune Hybrid Search', divider='rainbow')
    st.subheader('Note: The below parameters apply only when the Search type is set to Hybrid Search')
    weight = st.slider('Weight for Vector Search', 0.0, 1.0, 0.5,0.1,key = 'input_weight')
    st.selectbox('Select the Normalisation type',
    ('min_max',
    'l2'
    ),
   
    key = 'input_NormType',
    help = "Select the type of retriever\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers)"
    ) 
    st.selectbox('Select the Score Combination type',
    ('arithmetic_mean','geometric_mean','harmonic_mean'
    ),
   
    key = 'input_CombineType',
    help = "Select the type of retriever\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers)"
    )  


st.number_input("Number of Documents", min_value=1, max_value=5, value="min", step=1,  key='input_K', help=None)
    

    

st.markdown('---')


def write_user_message(md):
    col1, col2 = st.columns([0.60,12])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        #st.warning(md['question'])

        st.markdown("<div style='fontSize:25px;padding:3px 7px 3px 7px;borderWidth: 0px;background:#fffee0;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>"+md['question']+"</div>", unsafe_allow_html = True)
       



def render_answer(answer,search_type,index):

    # cols = st.columns(len(answer), gap = "large")
    
    # for i, x in enumerate(cols):
        
    #     x.image(answer[i]['image_url'],width = 100,caption=answer[i]['desc'])#caption=ans['desc']

    col_1, col_2, col_3 = st.columns([60,10,30])
    i = 0
    for ans in answer:
        urllib.request.urlretrieve( ans['image_url'], str(i)+".png") 
        img = Image.open(str(i)+".png")
        newsize = (200, 200)
        im1 = img.resize(newsize)#((int(img.size[0]*0.2),int(img.size[1]*0.2)))
        with col_1:
            st.image(im1)
            st.markdown(ans['desc'])
        i = i+1
    with col_2:
        st.write("")
    with col_3:
        if(index == len(st.session_state.questions)):

            rdn_key = ''.join([random.choice(string.ascii_letters)
                              for _ in range(10)])
            currentValue = st.session_state.input_searchType+str(st.session_state.input_weight)+st.session_state.input_NormType+st.session_state.input_CombineType+str(st.session_state.input_K)
            oldValue = st.session_state.inputs_["searchType"]+str(st.session_state.inputs_["weight"])+st.session_state.inputs_["NormType"]+st.session_state.inputs_["CombineType"]+str(st.session_state.inputs_["K"])

            def on_button_click():
                if(currentValue!=oldValue):
                    st.session_state.input_text = st.session_state.questions[-1]["question"]
                    st.session_state.answers.pop()
                    st.session_state.questions.pop()
                    
                    handle_input()
                    with placeholder.container():
                        render_all()

            if("currentValue"  in st.session_state):
                del st.session_state["currentValue"]

            try:
                del regenerate
            except:
                pass  

            print("------------------------")
            print(st.session_state)

            placeholder__ = st.empty()
            
            placeholder__.button("🔄",key=rdn_key,on_click=on_button_click, help = "This will regenerate the last response with new settings that you entered, Note: This applies to only the last response and to see difference in responses, you should change any of the settings above")#,type="primary",use_container_width=True)
     
        

    
    # with col2:
    #     # chat_box=st.empty() 
    #     # self.text+=token+"/" 
    #     # self.container.markdown(self.text) 
    #     #st.markdown(answer,unsafe_allow_html = True)
    #     for ans in answer:
    #         #st.markdown("<div style='padding:3px 7px 3px 7px;borderWidth: 0px;background:#D4F1F4;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'><b>"+answer+"</b></div>", unsafe_allow_html = True)
    #         st.image(ans['image_url'], caption=ans['desc'],use_column_width = 'always')
    # with col3:
    #     if(search_type== 'Conversational Search (RAG)'):
    #         st.markdown("<p style='padding:0px 5px 0px 5px;borderWidth: 0px;background:#D4F1F4;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 5px;'><b>RAG</b></p>", unsafe_allow_html = True, help = "Retriever type of the response")
    #     if(search_type== 'OpenSearch vector search'):
    #         st.markdown("<p style='padding:0px 5px 0px 5px;borderWidth: 0px;background:#D4F1F4;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 5px;'><b>OpenSearch</b></p>", unsafe_allow_html = True, help = "Retriever type of the response")
    #     if(search_type== 'LLM Text Generation'):
    #         st.markdown("<p style='padding:0px 5px 0px 5px;borderWidth: 0px;background:#D4F1F4;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 5px;'><b>LLM</b></p>", unsafe_allow_html = True, help = "Retriever type of the response")
        
    #     print("------------------------")
    #     print(type(st.session_state))
    #     print("------------------------")
    #     print(st.session_state)
    #     print("------------------------")
        
    # with col4:
    #     if(index == len(st.session_state.questions)):

    #         rdn_key = ''.join([random.choice(string.ascii_letters)
    #                           for _ in range(10)])
    #         currentValue = st.session_state.input_searchType+st.session_state.input_weight+st.session_state.input_NormType+st.session_state.input_CombineType+st.session_state.input_K
    #         oldValue = st.session_state.inputs_["searchType"]+st.session_state.inputs_["weight"]+st.session_state.inputs_["NormType"]+st.session_state.inputs_["CombineType"]+st.session_state.input_["K"]

    #         def on_button_click():
    #             if(currentValue!=oldValue):
    #                 st.session_state.input_text = st.session_state.questions[-1]["question"]
    #                 st.session_state.answers.pop()
    #                 st.session_state.questions.pop()
                    
    #                 handle_input()
    #                 with placeholder.container():
    #                     render_all()

    #         if("currentValue"  in st.session_state):
    #             del st.session_state["currentValue"]

    #         try:
    #             del regenerate
    #         except:
    #             pass  

    #         print("------------------------")
    #         print(st.session_state)

    #         placeholder__ = st.empty()
            
    #         placeholder__.button("🔄",key=rdn_key,on_click=on_button_click, help = "This will regenerate the last response with new settings that you entered, Note: This applies to only the last response and to see difference in responses, you should change any of the settings above")#,type="primary",use_container_width=True)
     
#Each answer will have context of the question asked in order to associate the provided feedback with the respective question
def write_chat_message(md, q,index):
    if('body' in md['answer']):
        res = json.loads(md['answer']['body'])
    else:
        res = md['answer']
    st.session_state['session_id'] = "1234"
    chat = st.container()
    with chat:
        render_answer(res,md['search_type'],index)
    
def render_all():  
    index = 0
    for (q, a) in zip(st.session_state.questions, st.session_state.answers):
        index = index +1
        print("answers----")
        print(a)
        write_user_message(q)
        write_chat_message(a, q,index)

placeholder = st.empty()
with placeholder.container():
  render_all()

st.markdown("")

