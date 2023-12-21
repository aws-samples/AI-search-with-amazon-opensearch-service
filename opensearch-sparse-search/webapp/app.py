import streamlit as st
import uuid
import os
import boto3
import requests
import api
from boto3 import Session
import botocore.session
import json
import random
import string
from PIL import Image 
import urllib.request 
import base64
import shutil
#from langchain.callbacks.base import BaseCallbackHandler


USER_ICON = "/home/ec2-user/images/user.png"
AI_ICON = "/home/ec2-user/images/opensearch-twitter-card.png"
REGENERATE_ICON = "/home/ec2-user/images/regenerate.png"
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
    print("***")
    print(model_type)
    print(st.session_state.img_doc)
    print(type(st.session_state.img_doc))
    if(st.session_state.img_doc is not None ):#and st.session_state.input_searchType == 'Multi-modal Search'):
        print("perform multimodal search")
        st.session_state.input_imageUpload = 'yes'
        Image.MAX_IMAGE_PIXELS = 100000000
        
        width = 2048
        height = 2048
        uploaded_images = os.path.join("/home/ec2-user/", "uploaded_images")
        if not os.path.exists(uploaded_images):
            os.mkdir(uploaded_images)

        with open(os.path.join("/home/ec2-user/uploaded_images",st.session_state.img_doc.name),"wb") as f: 
            f.write(st.session_state.img_doc.getbuffer())  
        photo = "/home/ec2-user/uploaded_images/"+st.session_state.img_doc.name
        with Image.open(photo) as image:
            image.verify()

        with Image.open(photo) as image:    
            if image.format.upper() in ["JPEG", "PNG","JPG"]:
                file_type = st.session_state.img_doc.name.split(".")[1]
                path = image.filename.rsplit(".", 1)[0]
                print(path)
                image.thumbnail((width, height))
                image.save(f"{path}-resized.{file_type}")
                width_ = 200
                height_ = 200
                image.thumbnail((width_, height_))
                image.save(f"{path}-resized_display.{file_type}")


        with open(photo.split(".")[0]+"-resized."+file_type, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode("utf8")
            st.session_state.input_image = input_image
    else:
        print("no image uploaded")
        st.session_state.input_imageUpload = 'no'
        st.session_state.input_image = ''


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
        'answer': api.call(json.dumps(inputs), st.session_state['session_id']),
        'search_type':inputs['searchType'],
        'id': len(st.session_state.questions)
    })
    #st.session_state.input_text=""
    #st.session_state.input_searchType=st.session_state.input_searchType

def write_top_bar():
    col1, col2,col3  = st.columns([1.75,20,5])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        #st.markdown("")
        input = st.text_input( "Ask here",label_visibility = "collapsed",key="input_text")
    with col3:
        clear = st.button("Clear")

    
        
    col4, col5  = st.columns([90,10])

    with col4:
        st.session_state.img_doc = st.file_uploader(
        "Upload the image and set the search type as \"Multi-modal Search\"", accept_multiple_files=False,type = ['png', 'jpg'])

    with col5:
        #hidden = st.button("RUN",disabled=True,key = "hidden")
        play = st.button("GO",on_click=handle_input,key = "play")
    

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

col1, col3 = st.columns([80,20])

with col1:
    search_type = st.selectbox('Select the Search type',
    ('Keyword Search',
    'Vector Search', 
    'Hybrid Search',
    'Multi-modal Search (Titan Multimodal Embeddings)'
    ),
   
    key = 'input_searchType',
    help = "Select the type of retriever\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers)"
    )

with col3:
    st.number_input("No. of docs", min_value=1, max_value=50, value=5, step=5,  key='input_K', help=None)
 

    

with st.sidebar:
    st.header('Fine-tune keyword Search', divider='rainbow')
    st.subheader('Note: The below selection applies only when the Search type is set to Keyword Search')
    sparse = st.checkbox('Expand query and documents with Sparse features', key = 'sparse')
    st.session_state.input_sparse = 'disabled'

    if sparse:
        #st.write(st.session_state.inputs_)
        st.session_state.input_sparse = 'enabled'

    st.markdown('---')
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

    st.markdown('---')

    st.header('Select the ML Model', divider='rainbow')
    st.subheader('Note: The below selection applies only when the Search type is set to Vector or Hybrid Search')
    
    model_type = st.selectbox('Select the Text Embeddings Model',
    ('GPT-J-6B (Sagemaker)',
    'Titan-Embed-Text-v1 (Bedrock)'
    ),
   
    key = 'input_modelType',
    help = "Select the type of retriever\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers)"
    )

   

    

st.markdown('---')


def write_user_message(md):
    col1, col2, col3 = st.columns([4,20,40])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        #st.warning(md['question'])

        st.markdown("<div style='fontSize:15px;padding:3px 7px 3px 7px;borderWidth: 0px;background:#ffffff;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>Input Text: </div><div style='fontSize:25px;padding:3px 7px 3px 7px;borderWidth: 0px;background:#fffee0;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>"+md['question']+"</div>", unsafe_allow_html = True)
    with col3:   
        st.markdown("<div style='fontSize:15px;padding:3px 7px 3px 7px;borderWidth: 0px;background:#ffffff;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>Input Image: </div>", unsafe_allow_html = True)
   
        if(st.session_state.input_imageUpload == 'yes'):
            st.image("/home/ec2-user/uploaded_images/"+st.session_state.img_doc.name.split(".")[0]+"-resized_display."+st.session_state.img_doc.name.split(".")[1])
        else:
            st.markdown("<div style='fontSize:15px;padding:3px 7px 3px 7px;borderWidth: 0px;background:#ffffff;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>None</div>", unsafe_allow_html = True)
   

    st.markdown('---')
        



def render_answer(answer,search_type,index):
    st.markdown("<div style='fontSize:25px;padding:3px 7px 3px 7px;borderWidth: 0px;background:#fffee0;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>Results</div>", unsafe_allow_html = True)
        
    if os.path.isdir("res_images"):
        shutil.rmtree('/home/ec2-user/res_images')
        #os.rmdir("res_images")
    
    res_images = os.path.join("/home/ec2-user/", "res_images")
    if not os.path.exists(res_images):
        os.mkdir(res_images)

    # cols = st.columns(len(answer), gap = "large")
    
    # for i, x in enumerate(cols):
        
    #     x.image(answer[i]['image_url'],width = 100,caption=answer[i]['desc'])#caption=ans['desc']

    placeholder_no_results  = st.empty()

    col_1, col_2, col_3 = st.columns([60,10,30])
    i = 0
    filter_out = 0
    for ans in answer:

        

        if('b5/b5319e00' in ans['image_url'] ):
            filter_out+=1
            continue

        
        # imgdata = base64.b64decode(ans['image_binary'])
        format_ = ans['image_url'].split(".")[-1]
       
        #urllib.request.urlretrieve(ans['image_url'], "/home/ec2-user/res_images/"+str(i)+"_."+format_) 

        
        Image.MAX_IMAGE_PIXELS = 100000000
        
        width = 500
        height = 500
          
        photo = "/home/ec2-user/res_images/"+str(i)+"_."+format_
        # with Image.open(photo) as image:
        #     image.verify()

        # with Image.open(photo) as image:    
        #     image.thumbnail((width, height))
        #     image.save("/home/ec2-user/res_images/"+str(i)+"_resized."+format_)


        with col_1:
            st.image(ans['image_url'], caption=ans['desc'])
        i = i+1
    with col_2:
        st.write("")
    with col_3:
        if(index == len(st.session_state.questions)):

            rdn_key = ''.join([random.choice(string.ascii_letters)
                              for _ in range(10)])
            currentValue = st.session_state.input_searchType+st.session_state.input_modelType+st.session_state.input_imageUpload+str(st.session_state.input_weight)+st.session_state.input_NormType+st.session_state.input_CombineType+str(st.session_state.input_K)+st.session_state.input_sparse
            oldValue = st.session_state.inputs_["searchType"]+st.session_state.inputs_["modelType"]+st.session_state.inputs_["imageUpload"]+str(st.session_state.inputs_["weight"])+st.session_state.inputs_["NormType"]+st.session_state.inputs_["CombineType"]+str(st.session_state.inputs_["K"])+st.session_state.inputs_["sparse"]

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
     
    if(filter_out > 0):
        placeholder_no_results.text(str(filter_out)+" result(s) removed due to missing or in-appropriate content")    

    
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

