import streamlit as st
import math
import io
import uuid
import os
import sys
import boto3
import requests
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/semantic_search")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/RAG")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/utilities")
from boto3 import Session
from pathlib import Path    
import botocore.session
#import os_index_df_sql
import json
import random
import string
from PIL import Image 
import urllib.request 
import base64
import shutil
import re
import utilities.re_ranker as re_ranker
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
import query_rewrite
import amazon_rekognition
#from st_click_detector import click_detector
import llm_eval
import all_search_execute


st.set_page_config(
    #page_title="Semantic Search using OpenSearch",
    #layout="wide",
    page_icon="images/opensearch_mark_default.png"
)
parent_dirname = "/".join((os.path.dirname(__file__)).split("/")[0:-1])
st.markdown("""
    <style>
    [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{
        gap: 0rem;
    }
    [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: 0rem;
    }
    </style>
    """,unsafe_allow_html=True)
#ps = PorterStemmer()

bedrock_ = boto3.client('bedrock-runtime',region_name='us-east-1')
#from langchain.callbacks.base import BaseCallbackHandler
search_all_type = True
if(search_all_type==True):
    search_types = ['Keyword Search',
    'Vector Search', 
    'Multimodal Search',
    'NeuralSparse Search',
    ]
else:
    search_types = ['Keyword Search',
   # 'Vector Search', 
    #'Hybrid Search',
    'Multimodal Search'
    ]

USER_ICON = "images/user.png"
AI_ICON = "images/opensearch-twitter-card.png"
REGENERATE_ICON = "images/regenerate.png"
IMAGE_ICON = "images/Image_Icon.png"
TEXT_ICON = "images/text.png"
s3_bucket_ = "pdf-repo-uploads"
            #"pdf-repo-uploads"

# Check if the user ID is already stored in the session state
if 'user_id' in st.session_state:
    user_id = st.session_state['user_id']
    print(f"User ID: {user_id}")

# If the user ID is not yet stored in the session state, generate a random UUID
# else:
#     user_id = str(uuid.uuid4())
#     st.session_state['user_id'] = user_id
#     dynamodb = boto3.resource('dynamodb')
#     table = dynamodb.Table('ml-search')
    


if 'session_id' not in st.session_state:
    st.session_state['session_id'] = ""
    
if 'input_reranker' not in st.session_state:
    st.session_state['input_reranker'] = "None"#"Cross Encoder"
    
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
    
if "clear_" not in st.session_state:
    st.session_state.clear_ = False
    
 
if "radio_disabled" not in st.session_state:
    st.session_state.radio_disabled = True

if "input_rad_1" not in st.session_state:
    st.session_state.input_rad_1 = ""



if "input_is_sql_query" not in st.session_state:
    st.session_state.input_is_sql_query = ""
    
if "input_sql_query" not in st.session_state:
    st.session_state.input_sql_query = ""
if "input_rewritten_query" not in st.session_state:
    st.session_state.input_rewritten_query = ""

if "input_hybridType" not in st.session_state:
    st.session_state.input_hybridType = "OpenSearch Hybrid Query"

if "ndcg_increase" not in st.session_state:
    st.session_state.ndcg_increase = " ~ "
    
if "inputs_" not in st.session_state:
    st.session_state.inputs_ = {}
    
if "img_container" not in st.session_state:
    st.session_state.img_container = ""

if "input_rekog_directoutput" not in st.session_state:
    st.session_state.input_rekog_directoutput = {}

if "input_weightage" not in st.session_state:
    st.session_state.input_weightage = {}   

if "img_gen" not in st.session_state:
    st.session_state.img_gen = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "answers_none_rank" not in st.session_state:
    st.session_state.answers_none_rank = []


if "input_text" not in st.session_state:
    st.session_state.input_text="trendy footwear for women"
    
if "input_ndcg" not in st.session_state:
    st.session_state.input_ndcg=0.0  

if "gen_image_str" not in st.session_state:
    st.session_state.gen_image_str=""

if "input_searchType" not in st.session_state:
    st.session_state.input_searchType = ["Keyword Search"]

if "input_NormType" not in st.session_state:
    st.session_state.input_NormType = "min_max"

if "input_CombineType" not in st.session_state:
    st.session_state.input_CombineType = "arithmetic_mean"

if "input_sparse" not in st.session_state:
    st.session_state.input_sparse = "disabled"
    
if "input_evaluate" not in st.session_state:
    st.session_state.input_evaluate = "disabled"
    
if "input_is_rewrite_query" not in st.session_state:
    st.session_state.input_is_rewrite_query = "disabled"
    
    
if "input_rekog_label" not in st.session_state:
    st.session_state.input_rekog_label = ""

if "input_modelType" not in st.session_state:
    st.session_state.input_modelType = "Titan-Embed-Text-v1"

if "input_weight" not in st.session_state:
    st.session_state.input_weight = 0.5

if "image_prompt2" not in st.session_state:
    st.session_state.image_prompt2 = ""

if "image_prompt" not in st.session_state:
    st.session_state.image_prompt = ""
    
if "bytes_for_rekog" not in st.session_state:
    st.session_state.bytes_for_rekog = ""
#bytes_for_rekog = ""


def generate_images(tab,inp_):
        #write_top_bar()
 
        request = json.dumps(
                    {
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": {"text": st.session_state.image_prompt},
                        "imageGenerationConfig": {
                            "numberOfImages": 3,
                            "quality": "standard",
                            "cfgScale": 8.0,
                            "height": 512,
                            "width": 512,
                        # "seed": seed,
                        },
                    }
                )

        if(inp_!=st.session_state.image_prompt):
            print("call bedrocck")
            response = bedrock_.invoke_model(
            modelId="amazon.titan-image-generator-v1", body=request
            )
            
            response_body = json.loads(response["body"].read())
            st.session_state.img_gen = response_body["images"]
        gen_images_dir = os.path.join(parent_dirname, "gen_images")
        if os.path.exists(gen_images_dir):
            shutil.rmtree(gen_images_dir)
        os.mkdir(gen_images_dir)
        width_ = 200
        height_ = 200
        index_ = 0
        #if(inp_!=st.session_state.image_prompt):
        
        if(len(st.session_state.img_gen)==0 and st.session_state.clear_ == True):
            #write_top_bar()
            placeholder1 = st.empty()
            with tab:
                with placeholder1.container():
                    st.empty()

        
        images_dis = []
        for image_ in st.session_state.img_gen:
            st.session_state.radio_disabled  = False
            if(index_==0):
                # with tab:
                #     rad1, rad2,rad3  = st.columns([98,1,1])
                # if(st.session_state.input_rad_1 is None):
                #     rand_ = ""
                # else:
                #     rand_ = st.session_state.input_rad_1
                # if(inp_!=st.session_state.image_prompt+rand_):
                #     with rad1:
                #         sel_rad_1 = st.radio("Choose one image", ["1","2","3"],index=None, horizontal = True,key = 'input_rad_1')

                with tab:
                    #sel_image = st.radio("", ["1","2","3"],index=None, horizontal = True)
                    if(st.session_state.img_container!=""):
                        st.session_state.img_container.empty()
                    place_ = st.empty()
                    img1, img2,img3  = place_.columns([30,30,30])
                    st.session_state.img_container = place_
                img_arr = [img1, img2,img3]
            
            base64_image_data = image_

            #st.session_state.gen_image_str = base64_image_data

            print("perform multimodal search")
        
            Image.MAX_IMAGE_PIXELS = 100000000
            filename = st.session_state.image_prompt+"_gen_"+str(index_)
            photo = parent_dirname+"/gen_images/"+filename+'.jpg'  # I assume you have a way of picking unique filenames
            imgdata = base64.b64decode(base64_image_data)
            with open(photo, 'wb') as f:
                f.write(imgdata) 

            
            
            with Image.open(photo) as image:    
                file_type = 'jpg'
                path = image.filename.rsplit(".", 1)[0]
                image.thumbnail((width_, height_))
                image.save(parent_dirname+"/gen_images/"+filename+"-resized_display."+file_type)

            with img_arr[index_]:
                placeholder_ = st.empty()
                placeholder_.image(parent_dirname+"/gen_images/"+filename+"-resized_display."+file_type)

            index_ = index_ + 1
      


def handle_input():
    if("text" in st.session_state.inputs_):
        if(st.session_state.inputs_["text"] != st.session_state.input_text):
            st.session_state.input_ndcg=0.0
    st.session_state.bytes_for_rekog = ""
    print("***")
    
    if(st.session_state.img_doc is not None or (st.session_state.input_rad_1 is not None and st.session_state.input_rad_1!="") ):#and st.session_state.input_searchType == 'Multi-modal Search'):
        print("perform multimodal search")
        st.session_state.input_imageUpload = 'yes'
        if(st.session_state.input_rad_1 is not None and st.session_state.input_rad_1!=""):
            
            num_str = str(int(st.session_state.input_rad_1.strip())-1)
            with open(parent_dirname+"/gen_images/"+st.session_state.image_prompt+"_gen_"+num_str+"-resized_display.jpg", "rb") as image_file:
                input_image = base64.b64encode(image_file.read()).decode("utf8")
                st.session_state.input_image = input_image
        
            if(st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType):
                st.session_state.bytes_for_rekog = Path(parent_dirname+"/gen_images/"+st.session_state.image_prompt+"_gen_"+num_str+".jpg").read_bytes()
        else:
            Image.MAX_IMAGE_PIXELS = 100000000
            width = 2048
            height = 2048
            uploaded_images = os.path.join(parent_dirname, "uploaded_images")

            if not os.path.exists(uploaded_images):
                os.mkdir(uploaded_images)

            with open(os.path.join(parent_dirname+"/uploaded_images",st.session_state.img_doc.name),"wb") as f: 
                f.write(st.session_state.img_doc.getbuffer())  
            photo = parent_dirname+"/uploaded_images/"+st.session_state.img_doc.name
            with Image.open(photo) as image:
                image.verify()

            with Image.open(photo) as image:  
                width_ = 200
                height_ = 200  
                if image.format.upper() in ["JPEG", "PNG","JPG"]:
                    path = image.filename.rsplit(".", 1)[0]
                    org_file_type = st.session_state.img_doc.name.split(".")[1]
                    image.thumbnail((width, height))
                    if(org_file_type.upper()=="PNG"):
                        file_type = "jpg"
                        image.convert('RGB').save(f"{path}-resized.{file_type}")
                    else:
                        file_type = org_file_type
                        image.save(f"{path}-resized.{file_type}")
                    
                    image.thumbnail((width_, height_))
                    image.save(f"{path}-resized_display.{org_file_type}")


            with open(photo.split(".")[0]+"-resized."+file_type, "rb") as image_file:
                input_image = base64.b64encode(image_file.read()).decode("utf8")
                st.session_state.input_image = input_image
                
            if(st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType):  
                st.session_state.bytes_for_rekog = Path(parent_dirname+"/uploaded_images/"+st.session_state.img_doc.name).read_bytes()
       
                
        
            
    else:
        print("no image uploaded")
        st.session_state.input_imageUpload = 'no'
        st.session_state.input_image = ''


    inputs = {}
    # if(st.session_state.input_imageUpload == 'yes'):
    #     st.session_state.input_searchType = 'Multi-modal Search'
    # if(st.session_state.input_sparse == 'enabled' or st.session_state.input_is_rewrite_query == 'enabled'):
    #     st.session_state.input_searchType = 'Keyword Search'
    if(st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType):
        st.session_state.input_rekog_label = amazon_rekognition.extract_image_metadata(st.session_state.bytes_for_rekog)
        if(st.session_state.input_text == ""):
            st.session_state.input_text = st.session_state.input_text + st.session_state.input_rekog_label
            
    # if(st.session_state.input_imageUpload == 'yes'):
    #     if(st.session_state.input_searchType!='Multi-modal Search'):
    #         if(st.session_state.input_searchType=='Keyword Search'):
    #             if(st.session_state.input_rekognition != 'enabled'):
    #                 st.error('For Keyword Search using images, enable "Enrich metadata for Images" in the left panel',icon = "🚨")
    #                 #st.session_state.input_rekognition = 'enabled'
    #                 st.switch_page('pages/1_Semantic_Search.py')
    #                 #st.stop()
                    
    #         else:
    #             st.error('Please set the search type as "Keyword Search (enabling Enrich metadata for Images) or Multi-modal Search"',icon = "🚨")
    #             #st.session_state.input_searchType='Multi-modal Search'
    #             st.switch_page('pages/1_Semantic_Search.py')
    #             #st.stop()
                

    weightage = {}
    st.session_state.weights_ = []
    total_weight = 0.0
    counter = 0
    num_search = len(st.session_state.input_searchType)
    any_weight_zero = False
    for type in st.session_state.input_searchType:
        key_weight = "input_"+type.split(" ")[0]+"-weight"
        total_weight = total_weight + st.session_state[key_weight]
        if(st.session_state[key_weight]==0):
            any_weight_zero = True
    print(total_weight)
    for key in st.session_state:
        
        if(key.startswith('input_')):
            original_key = key.removeprefix('input_')
            if('weight' not in key):
                inputs[original_key] = st.session_state[key]
            else:
                if(original_key.split("-")[0] + " Search" in st.session_state.input_searchType):
                    counter = counter +1
                    if(total_weight!=100 or any_weight_zero == True):
                        extra_weight = 100%num_search
                        if(counter == num_search):
                            cal_weight = math.trunc(100/num_search)+extra_weight
                        else:
                            cal_weight = math.trunc(100/num_search)
                            
                        st.session_state[key] = cal_weight
                        weightage[original_key] = cal_weight
                        st.session_state.weights_.append(cal_weight)
                    else:
                        weightage[original_key] = st.session_state[key]
                        st.session_state.weights_.append(st.session_state[key])
                else:
                    weightage[original_key] = 0.0
                    st.session_state[key] = 0.0
                    
        
   
                        
                        
                        
                    
              
        
        
                

                
    inputs['weightage']=weightage
    st.session_state.input_weightage = weightage
    
    print("====================")
    print(st.session_state.weights_)
    print(st.session_state.input_weightage )
    print("====================")
        #print("***************************")
        #print(sum(weights_))
        # if(sum(st.session_state.weights_)!=100):
        #     st.warning('The total weight of selected search type(s) should be equal to 100',icon = "🚨")
        #     refresh = st.button("Re-Enter")
        #     if(refresh):
        #         st.switch_page('pages/1_Semantic_Search.py')
        #         st.stop()
            
                
            #         #st.session_state.input_rekognition = 'enabled'
        #     st.rerun()
        
        
            
    st.session_state.inputs_ = inputs
    
    #st.write(inputs) 
    question_with_id = {
        'question': inputs["text"],
        'id': len(st.session_state.questions)
    }
    st.session_state.questions = []
    st.session_state.questions.append(question_with_id)
    
    st.session_state.answers = []
    
    if(st.session_state.input_is_sql_query == 'enabled'):
        os_index_df_sql.sql_process(st.session_state.input_text)
        print(st.session_state.input_sql_query)
    else:
        st.session_state.input_sql_query = ""
        
    
    if(st.session_state.input_is_rewrite_query == 'enabled' or (st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType)):
        query_rewrite.get_new_query_res(st.session_state.input_text)
        print("-------------------")
        print(st.session_state.input_rewritten_query)
        print("-------------------")
    else:
        st.session_state.input_rewritten_query = ""
        
    # elif(st.session_state.input_rekog_label!="" and st.session_state.input_rekognition == 'enabled'):
    #     ans__ = amazon_rekognition.call(st.session_state.input_text,st.session_state.input_rekog_label)
    # else:
    ans__ = all_search_execute.handler(inputs, st.session_state['session_id'])
    
    st.session_state.answers.append({
        'answer': ans__,#all_search_api.call(json.dumps(inputs), st.session_state['session_id']),
        'search_type':inputs['searchType'],
        'id': len(st.session_state.questions)
    })
    
    st.session_state.answers_none_rank = st.session_state.answers
    if(st.session_state.input_reranker == "None"):
        st.session_state.answers = st.session_state.answers_none_rank 
    else:
        st.session_state.answers = re_ranker.re_rank("search",st.session_state.input_reranker,st.session_state.input_searchType,st.session_state.questions, st.session_state.answers)
    if(st.session_state.input_evaluate) == "enabled":
        llm_eval.eval(st.session_state.questions, st.session_state.answers)
    #st.session_state.input_text=""
    #st.session_state.input_searchType=st.session_state.input_searchType

def write_top_bar():
    # st.markdown("""
    # <style>
    # [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
    #     gap: 0rem;
    # }
    # </style>
    # """,unsafe_allow_html=True)
    #print("top bar")
    # st.title(':mag: AI powered OpenSearch')
    # st.write("")
    # st.write("")
    col1, col2,col3,col4  = st.columns([2.5,35,7,6])
    with col1:
        st.image(TEXT_ICON, use_column_width='always')
    with col2:
        #st.markdown("")
        input = st.text_input( "Ask here",label_visibility = "collapsed",key="input_text",placeholder = "Type your query")
    with col3:
        play = st.button("SEARCH",on_click=handle_input,key = "play")
        
    with col4:
        clear = st.button("Clear")
    
    col5, col6  = st.columns([4.5,90])

    with col5:
        st.image(IMAGE_ICON, use_column_width='always')
    with col6:   
        with st.expander(':green[Search by using an image]'):
            tab2, tab1 = st.tabs(["Upload Image","Generate Image by AI"])
            
            with tab1:
                c1,c2 = st.columns([80,20])
                with c1:
                    gen_images=st.text_area("Text to Image:",placeholder = "Enter the text prompt to generate images",height = 50, key = "image_prompt")
                with c2:
                    st.markdown("<div style = 'height:50px'></div>",unsafe_allow_html=True)
                    st.button("Generate image",disabled=False,key = "generate",on_click = generate_images, args=(tab1,"default_img"))
                
                # image_select = st.select_slider(
                #     "Select a image",
                #     options=["Image 1","Image 2","Image 3"], value = None, disabled = st.session_state.radio_disabled,key = "image_select")
                image_select = st.radio("Choose one image", ["Image 1","Image 2","Image 3"],index=None, horizontal = True,key = 'image_select',disabled = st.session_state.radio_disabled)
                st.markdown("""
                            <style>
                            [role=radiogroup]{
                                gap: 10rem;
                            }
                            </style>
                            """,unsafe_allow_html=True)
                if(st.session_state.image_select is not None and st.session_state.image_select !="" and len(st.session_state.img_gen)!=0):
                    print("image_select")
                    print("------------")
                    print(st.session_state.image_select)
                    st.session_state.input_rad_1 = st.session_state.image_select.split(" ")[1]
                else:
                    st.session_state.input_rad_1 = ""
                # rad1, rad2,rad3  = st.columns([33,33,33])
                # with rad1:
                #     btn1 = st.button("choose image 1", disabled = st.session_state.radio_disabled)
                # with rad2:
                #     btn2 = st.button("choose image 2", disabled = st.session_state.radio_disabled)
                # with rad3:
                #     btn3 = st.button("choose image 3", disabled = st.session_state.radio_disabled)
                # if(btn1):
                #     st.session_state.input_rad_1 = "1" 
                # if(btn2):
                #     st.session_state.input_rad_1 = "2" 
                # if(btn3):
                #     st.session_state.input_rad_1 = "3" 


        generate_images(tab1,gen_images)   
            
            
        with tab2:
            st.session_state.img_doc = st.file_uploader(
            "Upload image", accept_multiple_files=False,type = ['png', 'jpg'])
            
    
        
        

    return clear,tab1

clear,tab_ = write_top_bar()

if clear:
    
    
    print("clear1")
    st.session_state.questions = []
    st.session_state.answers = []
    
    st.session_state.clear_ = True
    st.session_state.image_prompt2 = ""
    st.session_state.input_rekog_label = ""
    
    st.session_state.radio_disabled = True
    
    if(len(st.session_state.img_gen)!=0):
        st.session_state.img_container.empty()
        st.session_state.img_gen = []
        st.session_state.input_rad_1 = ""
    
        
        # placeholder1 = st.empty()
        # with placeholder1.container():
        #     generate_images(tab_,st.session_state.image_prompt)
        
        
    #st.session_state.input_text=""
    # st.session_state.input_searchType="Conversational Search (RAG)"
    # st.session_state.input_temperature = "0.001"
    # st.session_state.input_topK = 200
    # st.session_state.input_topP = 0.95
    # st.session_state.input_maxTokens = 1024

col1, col3, col4 = st.columns([70,20,10])

with col1:
    

    search_type = st.multiselect('Select the Search type(s)',
   search_types,['Keyword Search'],
   
    key = 'input_searchType',
    help = "Select the type of Search, adding more than one search type will activate hybrid search"#\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers)"
    )

with col3:
    st.number_input("No. of docs", min_value=1, max_value=50, value=5, step=5,  key='input_K', help=None)
with col4:
    st.markdown("<p style='fontSize:14.5px'>Evaluate</p>",unsafe_allow_html=True)
    evaluate = st.toggle(' ', key = 'evaluate', disabled = False) #help = "Checking this box will use LLM to evaluate results as relevant and irrelevant. \n\n This option increases the latency")
    if(evaluate):
        st.session_state.input_evaluate = "enabled"
        #llm_eval.eval(st.session_state.questions, st.session_state.answers)
    else:
        st.session_state.input_evaluate = "disabled"
        

if(search_all_type == True):
    with st.sidebar:
        st.page_link("/home/ubuntu/AI-search-with-amazon-opensearch-service/OpenSearchApp/app.py", label=":orange[Home]", icon="🏠")
        #st.warning('Note: After changing any of the below settings, click "SEARCH" button or 🔄 to apply the changes', icon="⚠️")
        #st.header('     :gear: :orange[Fine-tune Search]')
        #st.write("Note: After changing any of the below settings, click 'SEARCH' button or '🔄' to apply the changes")
        st.subheader(':blue[Keyword Search]')

        rewrite_query = st.checkbox('Enrich Docs and Re-write query DSL', key = 'query_rewrite', disabled = False, help = "Checking this box will use LLM to rewrite your query. \n\n Here your natural language query is transformed into OpenSearch query with added filters and attributes")
        sql_query = st.checkbox('Re-write as SQL query', key = 'sql_rewrite', disabled = True, help = "In Progress")
        st.session_state.input_is_rewrite_query = 'disabled'
        st.session_state.input_is_sql_query = 'disabled'
        if rewrite_query:
            #st.write(st.session_state.inputs_)
            st.session_state.input_is_rewrite_query = 'enabled'
        if sql_query:
            #st.write(st.session_state.inputs_)
            st.session_state.input_is_sql_query = 'enabled'
        
        
        
        #st.markdown('---')
        #st.header('Fine-tune keyword Search', divider='rainbow')
        #st.subheader('Note: The below selection applies only when the Search type is set to Keyword Search')
           
         
        # st.markdown("<u>Enrich metadata for :</u>",unsafe_allow_html=True) 
        

        
        # c3,c4 = st.columns([10,90])
        # with c4:
        #     rekognition = st.checkbox('Images', key = 'rekognition', help = "Checking this box will use AI to extract metadata for images that are present in query and documents")
        # if rekognition:
        #     #st.write(st.session_state.inputs_)
        #     st.session_state.input_rekognition = 'enabled'
        # else:
        #     st.session_state.input_rekognition = "disabled"

        #st.markdown('---')
        #st.header('Fine-tune Hybrid Search', divider='rainbow')
        #st.subheader('Note: The below parameters apply only when the Search type is set to Hybrid Search')
        
        
        
        #st.write("---")
        st.subheader(':blue[Hybrid Search]')
        st.selectbox('Select the Hybrid Search type',
         ("OpenSearch Hybrid Query","Reciprocal Rank Fusion"),key = 'input_hybridType')
        # equal_weight = st.button("Give equal weights to selected searches")
        
                    
                
                
             
            
        #st.warning('Weight of each of the selected search type should be greater than 0 and the total weight of all the selected search type(s) should be equal to 100',icon = "⚠️")
            
        
        #st.markdown("<p style = 'font-size:14.5px;font-style:italic;'>Set Weights</p>",unsafe_allow_html=True)
        
        with st.expander("Set query Weightage:"):
            st.number_input("Keyword %", min_value=0, max_value=100, value=100, step=5,  key='input_Keyword-weight', help=None)
            st.number_input("Vector %", min_value=0, max_value=100, value=0, step=5,  key='input_Vector-weight', help=None)
            st.number_input("Multimodal %", min_value=0, max_value=100, value=0, step=5,  key='input_Multimodal-weight', help=None)
            st.number_input("NeuralSparse %", min_value=0, max_value=100, value=0, step=5,  key='input_NeuralSparse-weight', help=None)
        
        # if(equal_weight):
        #     counter = 0
        #     num_search = len(st.session_state.input_searchType)
        #     weight_type = ["input_Keyword-weight","input_Vector-weight","input_Multimodal-weight","input_NeuralSparse-weight"]
        #     for type in weight_type:
        #         if(type.split("-")[0].replace("input_","")+ " Search" in st.session_state.input_searchType):
        #             print("ssssssssssss")
        #             counter = counter +1
        #             extra_weight = 100%num_search
        #             if(counter == num_search):
        #                 cal_weight = math.trunc(100/num_search)+extra_weight
        #             else:
        #                 cal_weight = math.trunc(100/num_search)
        #             st.session_state[weight_type] = cal_weight
        #         else:
        #             st.session_state[weight_type] = 0
        #weight = st.slider('Weight for Vector Search', 0.0, 1.0, 0.5,0.1,key = 'input_weight', help = 'Use this slider to set the weightage for keyword and vector search, higher values of the slider indicate the increased weightage for semantic search.\n\n This applies only when the search type is set to Hybrid Search')
        # st.selectbox('Select the Normalisation type',
        # ('min_max',
        # 'l2'
        # ),
        #st.write("---")
        # key = 'input_NormType',
        # disabled = True,
        # help = "Select the type of Normalisation to be applied on the two sets of scores"
        # ) 

        # st.selectbox('Select the Score Combination type',
        # ('arithmetic_mean','geometric_mean','harmonic_mean'
        # ),
    
        # key = 'input_CombineType',
        # disabled = True,
        # help = "Select the Combination strategy to be used while combining the two scores of the two search queries for every document"
        # )  

        #st.markdown('---')

        #st.header('Select the ML Model for text embedding', divider='rainbow')
        #st.subheader('Note: The below selection applies only when the Search type is set to Vector or Hybrid Search')
        st.subheader(':blue[Re-ranking]')
        reranker = st.selectbox('Choose a Re-Ranker',
        ('None','Kendra Rescore','Cross Encoder'
        
        ),
        
        key = 'input_reranker',
        help = 'Select the Re-Ranker type, select "None" to apply no re-ranking of the results',
        #on_change = re_ranker.re_rank,
        args=(st.session_state.questions, st.session_state.answers)

        )
        # st.write("---")
        # st.subheader('Text Embeddings Model')
        # model_type = st.selectbox('Select the Text Embeddings Model',
        # ('Titan-Embed-Text-v1','GPT-J-6B'
        
        # ),
    
        # key = 'input_modelType',
        # help = "Select the Text embedding model, this applies only for the vector and hybrid search"
        # )

        #st.markdown('---')

        

        

    

st.markdown('---')


def write_user_message(md,ans):
    #print(ans)
    ans = ans["answer"][0]
    col1, col2, col3 = st.columns([3,40,20])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        #st.warning(md['question'])
        st.markdown("<div style='fontSize:15px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>Input Text: </div><div style='fontSize:25px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;font-style: italic;color:#e28743'>"+md['question']+"</div>", unsafe_allow_html = True)
        if('query_sparse' in ans):
            with st.expander("Expanded Query:"):
                query_sparse = dict(sorted(ans['query_sparse'].items(), key=lambda item: item[1],reverse=True))
                filtered_query_sparse = dict()
                for key in query_sparse:
                    if(query_sparse[key]>=0.5):
                        filtered_query_sparse[key] = round(query_sparse[key], 2)
                st.write(filtered_query_sparse)
        if(st.session_state.input_is_rewrite_query == "enabled" and st.session_state.input_rewritten_query !=""):
            with st.expander("Re-written Query:"):
                st.json(st.session_state.input_rewritten_query,expanded = True)
                
            
    with col3:   
        st.markdown("<div style='fontSize:15px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>Input Image: </div>", unsafe_allow_html = True)
   
        if(st.session_state.input_imageUpload == 'yes'):

            if(st.session_state.input_rad_1 is not None and st.session_state.input_rad_1!=""):
                num_str = str(int(st.session_state.input_rad_1.strip())-1)
                img_file = parent_dirname+"/gen_images/"+st.session_state.image_prompt+"_gen_"+num_str+"-resized_display.jpg"
            else:
                img_file = parent_dirname+"/uploaded_images/"+st.session_state.img_doc.name.split(".")[0]+"-resized_display."+st.session_state.img_doc.name.split(".")[1]
    
            st.image(img_file)
            if(st.session_state.input_rekog_label !=""):
                with st.expander("Enriched Query Metadata:"):
                        st.markdown('<p>'+json.dumps(st.session_state.input_rekog_directoutput)+'<p>',unsafe_allow_html=True)
        else:
            st.markdown("<div style='fontSize:15px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>None</div>", unsafe_allow_html = True)
            
            
   

    st.markdown('---')
        

# def stem_(sentence):
#     words = word_tokenize(sentence)
    
#     words_stem = []

#     for w in words:
#         words_stem.append( ps.stem(w))
#     return words_stem

def render_answer(answer,index):
    column1, column2 = st.columns([6,90])
    with column1:
        st.image(AI_ICON, use_column_width='always')
    with column2:
        st.markdown("<div style='fontSize:25px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>Results </div>", unsafe_allow_html = True)
        if(st.session_state.input_evaluate == "enabled" and st.session_state.input_ndcg > 0):
            span_color = "white"
            if("&uarr;" in st.session_state.ndcg_increase):
                span_color = "green"
            if("&darr;" in st.session_state.ndcg_increase):
                span_color = "red"
            st.markdown("<span style='fontSize:20px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 20px;font-family:Courier New;color:#e28743'>Relevance:" +str('%.3f'%(st.session_state.input_ndcg)) + "</span><span style='font-size:30px;font-weight:bold;color:"+span_color+"'>"+st.session_state.ndcg_increase.split("~")[0] +"</span><span style='font-size:15px;font-weight:bold;font-family:Courier New;color:"+span_color+"'> "+st.session_state.ndcg_increase.split("~")[1]+"</span>", unsafe_allow_html = True)
        
            
            #st.markdown("<span style='font-size:30px;color:"+span_color+"'>"+st.session_state.ndcg_increase.split("~")[0] +"</span><span style='font-size:15px;font-family:Courier New;color:"+span_color+"'>"+st.session_state.ndcg_increase.split("~")[1]+"</span>",unsafe_allow_html = True)
        
    

    placeholder_no_results  = st.empty()

    col_1, col_2,col_3 = st.columns([70,10,20])
    i = 0
    filter_out = 0
    for ans in answer:

        

        if('b5/b5319e00' in ans['image_url'] ):
            filter_out+=1
            continue

        
        # imgdata = base64.b64decode(ans['image_binary'])
        format_ = ans['image_url'].split(".")[-1]
       
        #urllib.request.urlretrieve(ans['image_url'], "/home/ubuntu/res_images/"+str(i)+"_."+format_) 

        
        Image.MAX_IMAGE_PIXELS = 100000000
        
        width = 500
        height = 500
          
        

        with col_1:
            inner_col_1,inner_col_2 = st.columns([8,92])
            with inner_col_2:
                st.image(ans['image_url'].replace('https://retail-demo-store-us-east-1.s3.amazonaws.com/images/','/home/ubuntu/images_retail/'))

                if("highlight" in ans and 'Keyword Search' in st.session_state.input_searchType):
                    test_strs = ans["highlight"]
                    tag = "em"
                    res__ = []
                    for test_str in test_strs:
                        start_idx = test_str.find("<" + tag + ">")
                        
                        while start_idx != -1:
                            end_idx = test_str.find("</" + tag + ">", start_idx)
                            if end_idx == -1:
                                break
                            res__.append(test_str[start_idx+len(tag)+2:end_idx])
                            start_idx = test_str.find("<" + tag + ">", end_idx)

                        
                    desc__ = ans['desc'].split(" ")
                        
                    final_desc = "<p>"
                    
                    ###### stemming and highlighting
                    
                    # ans_text = ans['desc']
                    # query_text = st.session_state.input_text

                    # ans_text_stemmed = set(stem_(ans_text))
                    # query_text_stemmed = set(stem_(query_text))

                    # common = ans_text_stemmed.intersection( query_text_stemmed)
                    # #unique = set(document_1_words).symmetric_difference(  )

                    # desc__stemmed = stem_(desc__)

                    # for word_ in desc__stemmed:
                    #     if(word_ in common):


                    for word in desc__:
                        if(re.sub('[^A-Za-z0-9]+', '', word) in res__):
                            final_desc +=  "<span style='color:#e28743;font-weight:bold'>"+word+"</span> "
                        else:
                            final_desc += word + " "
                    
                    final_desc += "</p>"

                    st.markdown(final_desc,unsafe_allow_html = True)
                else:
                    st.write(ans['desc'])
                if("sparse" in ans):
                    with st.expander("Expanded document:"):
                        sparse_ = dict(sorted(ans['sparse'].items(), key=lambda item: item[1],reverse=True))
                        filtered_sparse = dict()
                        for key in sparse_:
                            if(sparse_[key]>=1.0):
                                filtered_sparse[key] = round(sparse_[key], 2)
                        st.write(filtered_sparse)
                with st.expander("Document Metadata:"):
                    if("rekog" in ans):
                        div_size = [50,50]
                    else:
                        div_size = [99,1]
                    div1,div2 = st.columns(div_size)
                    with div1:
                        st.write(":green[default:]")
                        st.json({"category":ans['category'],"price":ans['price'],"gender_affinity":ans['gender_affinity']},expanded = False)
                    with div2:
                        if("rekog" in ans):
                            st.write(":green[enriched:]")
                            st.json(ans['rekog'],expanded = False)
            with inner_col_1:
                
                if(st.session_state.input_evaluate == "enabled"):
                    with st.container(border = False):
                        if("relevant" in ans.keys()):
                            if(ans['relevant']==True):
                                st.write(":white_check_mark:")
                            else:
                                st.write(":x:")
                    
        i = i+1
    # with col_2:
    #     if(st.session_state.input_evaluate == "enabled"):
    #         st.markdown("<div style='fontSize:12px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;font-weight:bold;height: fit-content;border-radius: 20px;font-family:Courier New;color:#e28743'>DCG: " +str('%.3f'%(st.session_state.input_ndcg)) + "</div>", unsafe_allow_html = True)
    # with col_2_b:
    #     span_color = "white"
    #     if("&uarr;" in st.session_state.ndcg_increase):
    #         span_color = "green"
    #     if("&darr;" in st.session_state.ndcg_increase):
    #         span_color = "red"
    #     st.markdown("<span style='font-size:30px;color:"+span_color+"'>"+st.session_state.ndcg_increase.split("~")[0] +"</span><span style='font-size:15px;font-family:Courier New;color:"+span_color+"'>"+st.session_state.ndcg_increase.split("~")[1]+"</span>",unsafe_allow_html = True)
            
            
    with col_3:
        if(index == len(st.session_state.questions)):

            rdn_key = ''.join([random.choice(string.ascii_letters)
                              for _ in range(10)])
            currentValue = "".join(st.session_state.input_searchType)+st.session_state.input_imageUpload+json.dumps(st.session_state.input_weightage)+st.session_state.input_NormType+st.session_state.input_CombineType+str(st.session_state.input_K)+st.session_state.input_sparse+st.session_state.input_reranker+st.session_state.input_is_rewrite_query+st.session_state.input_evaluate+st.session_state.input_image+st.session_state.input_rad_1+st.session_state.input_reranker+st.session_state.input_hybridType
            oldValue = "".join(st.session_state.inputs_["searchType"])+st.session_state.inputs_["imageUpload"]+str(st.session_state.inputs_["weightage"])+st.session_state.inputs_["NormType"]+st.session_state.inputs_["CombineType"]+str(st.session_state.inputs_["K"])+st.session_state.inputs_["sparse"]+st.session_state.inputs_["reranker"]+st.session_state.inputs_["is_rewrite_query"]+st.session_state.inputs_["evaluate"]+st.session_state.inputs_["image"]+st.session_state.inputs_["rad_1"]+st.session_state.inputs_["reranker"]+st.session_state.inputs_["hybridType"]
            
            def on_button_click():
                if(currentValue!=oldValue):
                    st.session_state.input_text = st.session_state.questions[-1]["question"]
                    st.session_state.answers.pop()
                    st.session_state.questions.pop()
                    
                    handle_input()
                    #re_ranker.re_rank(st.session_state.questions, st.session_state.answers)
                    with placeholder.container():
                        render_all()
                
                        

            if("currentValue"  in st.session_state):
                del st.session_state["currentValue"]

            try:
                del regenerate
            except:
                pass  

            print("------------------------")
            #print(st.session_state)

            placeholder__ = st.empty()
            
            placeholder__.button("🔄",key=rdn_key,on_click=on_button_click, help = "This will regenerate the responses with new settings that you entered, Note: To see difference in responses, you should change any of the applicable settings")#,type="primary",use_container_width=True)
     
    if(filter_out > 0):
        placeholder_no_results.text(str(filter_out)+" result(s) removed due to missing or in-appropriate content")    

    
    
#Each answer will have context of the question asked in order to associate the provided feedback with the respective question
def write_chat_message(md, q,index):
    if('body' in md['answer']):
        res = json.loads(md['answer']['body'])
    else:
        res = md['answer']
    st.session_state['session_id'] = "1234"
    chat = st.container()
    with chat:
        render_answer(res,index)
    
def render_all():  
    index = 0
    for (q, a) in zip(st.session_state.questions, st.session_state.answers):
        index = index +1
        #print("answers----")
        #print(a)
        ans_ = st.session_state.answers[0]
        write_user_message(q,ans_)
        write_chat_message(a, q,index)

placeholder = st.empty()
with placeholder.container():
  render_all()
  
  #generate_images("",st.session_state.image_prompt)

st.markdown("")
