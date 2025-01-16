import streamlit as st
import math
import io
import uuid
import os
import sys
import boto3
import requests
from requests_aws4auth import AWS4Auth
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/semantic_search")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/RAG")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/utilities")
from boto3 import Session
from pathlib import Path    
import botocore.session
import subprocess
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
import dynamo_state as ds


st.set_page_config(
    #page_title="Semantic Search using OpenSearch",
    #layout="wide",
    page_icon="images/opensearch_mark_default.png"
)
parent_dirname = "/".join((os.path.dirname(__file__)).split("/")[0:-1])
st.markdown("""
        <style>
               .block-container {
                    padding-top: 2.75rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
#ps = PorterStemmer()

st.session_state.REGION = ds.get_region()


#from langchain.callbacks.base import BaseCallbackHandler


USER_ICON = "OpenSearchApp/images/user.png"
AI_ICON = "OpenSearchApp/images/opensearch-twitter-card.png"
REGENERATE_ICON = "OpenSearchApp/images/regenerate.png"
IMAGE_ICON = "OpenSearchApp/images/Image_Icon.png"
TEXT_ICON = "OpenSearchApp/images/text.png"
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
    
if "input_clear_filter" not in st.session_state:
    st.session_state.input_clear_filter = False
    
 
if "radio_disabled" not in st.session_state:
    st.session_state.radio_disabled = True

if "input_rad_1" not in st.session_state:
    st.session_state.input_rad_1 = ""

if "input_manual_filter" not in st.session_state:
    st.session_state.input_manual_filter = ds.get_from_dynamo("input_manual_filter")

if "input_category" not in st.session_state:
    st.session_state.input_category = None
    
if "input_gender" not in st.session_state:
    st.session_state.input_gender = None
    
if "input_imageUpload" not in st.session_state:
    st.session_state.input_imageUpload = 'no'
    
# if "input_price" not in st.session_state:
#     st.session_state.input_price = (0,0)
    
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
    st.session_state.input_text="black jacket for men"#"black jacket for men under 120 dollars"
    
if "input_ndcg" not in st.session_state:
    st.session_state.input_ndcg=0.0  

if "gen_image_str" not in st.session_state:
    st.session_state.gen_image_str=""

# if "input_searchType" not in st.session_state:
#     st.session_state.input_searchType = ['Keyword Search']
    
# if "input_must" not in st.session_state:
#     st.session_state.input_must = ["Category","Price","Gender","Style"]
    
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
    

if "input_sparse_filter" not in st.session_state:
    st.session_state.input_sparse_filter = 0.4

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
    
if "OpenSearchDomainEndpoint" not in st.session_state:
    st.session_state.OpenSearchDomainEndpoint = ds.get_from_dynamo("OpenSearchDomainEndpoint")
    
if "max_selections" not in st.session_state:
    st.session_state.max_selections = ds.get_from_dynamo("max_selections")
    
if "sagemaker_re_ranker" not in st.session_state:
    st.session_state.sagemaker_re_ranker = ds.get_from_dynamo("sagemaker_re_ranker")
    
if "neural_sparse_two_phase_search_pipeline" not in st.session_state:
    st.session_state.neural_sparse_two_phase_search_pipeline = ds.get_from_dynamo("neural_sparse_two_phase_search_pipeline")   
    
if "bedrock_re_ranker" not in st.session_state:
    st.session_state.bedrock_re_ranker = ds.get_from_dynamo("bedrock_re_ranker")

if "llm_search_request_pipeline" not in st.session_state:
    st.session_state.llm_search_request_pipeline = ds.get_from_dynamo("llm_search_request_pipeline")

host = 'https://'+st.session_state.OpenSearchDomainEndpoint+'/'
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, st.session_state.REGION, service, session_token=credentials.token)
headers = {"Content-Type": "application/json"}

###### check for search pipelines #######


###### 1. hybrid search pipeline #######
opensearch_search_pipeline = (requests.get(host+'_search/pipeline/hybrid_search_pipeline', auth=awsauth,headers=headers)).text
print("opensearch_search_pipeline")
print(opensearch_search_pipeline)
if(opensearch_search_pipeline!='{}'):
    st.session_state.max_selections = "None"
else:
    st.session_state.max_selections = "1"
        
ds.store_in_dynamo('max_selections',st.session_state.max_selections )

########
opensearch_sparse_search_pipeline = (requests.get(host+'_search/pipeline/neural_sparse_two_phase_search_pipeline', auth=awsauth,headers=headers)).text
if(opensearch_sparse_search_pipeline!='{}'):
    st.session_state.neural_sparse_two_phase_search_pipeline = opensearch_sparse_search_pipeline
else:
    st.session_state.neural_sparse_two_phase_search_pipeline = ""
        
ds.store_in_dynamo('neural_sparse_two_phase_search_pipeline',st.session_state.neural_sparse_two_phase_search_pipeline )
######

###### 2. sagemaker rerank search pipeline #######
opensearch_sagemaker_rerank_pipeline = (requests.get(host+'_search/pipeline/sagemaker_rerank_pipeline', auth=awsauth,headers=headers)).text
print("opensearch_sagemaker_rerank_pipeline")
print(opensearch_sagemaker_rerank_pipeline)
if(opensearch_sagemaker_rerank_pipeline!='{}'):
    st.session_state.sagemaker_re_ranker = "true"
    total_pipeline = json.loads(opensearch_sagemaker_rerank_pipeline)
    if('response_processors' in total_pipeline['sagemaker_rerank_pipeline'].keys()):
        st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID = total_pipeline['sagemaker_rerank_pipeline']['response_processors'][0]['rerank']['ml_opensearch']['model_id']
        ds.store_in_dynamo('SAGEMAKER_CrossEncoder_MODEL_ID',st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID )
        
else:
    st.session_state.sagemaker_re_ranker = "false"
ds.store_in_dynamo('sagemaker_re_ranker',st.session_state.sagemaker_re_ranker )

###### 3. Bedrock rerank search pipeline #######
opensearch_bedrock_rerank_pipeline = (requests.get(host+'_search/pipeline/bedrock_rerank_pipeline', auth=awsauth,headers=headers)).text
print("opensearch_bedrock_rerank_pipeline")
print(opensearch_bedrock_rerank_pipeline)
if(opensearch_bedrock_rerank_pipeline!='{}'):
    st.session_state.bedrock_re_ranker = "true"
    total_pipeline = json.loads(opensearch_bedrock_rerank_pipeline)
    if('response_processors' in total_pipeline['bedrock_rerank_pipeline'].keys()):
        st.session_state.BEDROCK_Rerank_MODEL_ID = total_pipeline['bedrock_rerank_pipeline']['response_processors'][0]['rerank']['ml_opensearch']['model_id']
        ds.store_in_dynamo('BEDROCK_Rerank_MODEL_ID',st.session_state.BEDROCK_Rerank_MODEL_ID )
        
else:
    st.session_state.bedrock_re_ranker = "false"
ds.store_in_dynamo('bedrock_re_ranker',st.session_state.bedrock_re_ranker )

###### 4. LLM request search pipeline (text) #######

opensearch_llm_search_request_pipeline = (requests.get(host+'_search/pipeline/LLM_search_request_pipeline', auth=awsauth,headers=headers)).text
print("LLM_search_request_pipeline")
print(opensearch_llm_search_request_pipeline)
if(opensearch_llm_search_request_pipeline!='{}'):
    st.session_state.llm_search_request_pipeline = "true"
    total_pipeline = json.loads(opensearch_llm_search_request_pipeline)
    if('request_processors' in total_pipeline['LLM_search_request_pipeline'].keys()):
        st.session_state.BEDROCK_Claude3_text_MODEL_ID  = total_pipeline['LLM_search_request_pipeline']['request_processors'][0]['ml_inference']['model_id']
        ds.store_in_dynamo('BEDROCK_Claude3_text_MODEL_ID', st.session_state.BEDROCK_Claude3_text_MODEL_ID )
else:
    st.session_state.llm_search_request_pipeline = "false"
ds.store_in_dynamo('llm_search_request_pipeline',st.session_state.llm_search_request_pipeline )

###### 5. LLM request search pipeline (image) #######

opensearch_llm_search_request_pipeline_image = (requests.get(host+'_search/pipeline/LLM_search_request_pipeline_image', auth=awsauth,headers=headers)).text
print("LLM_search_request_pipeline_image")
print(opensearch_llm_search_request_pipeline_image)
if(opensearch_llm_search_request_pipeline_image!='{}'):
    st.session_state.llm_search_request_pipeline_image = "true"
    total_pipeline = json.loads(opensearch_llm_search_request_pipeline_image)
    if('request_processors' in total_pipeline['LLM_search_request_pipeline_image'].keys()):
        st.session_state.BEDROCK_Claude3_image_MODEL_ID  = total_pipeline['LLM_search_request_pipeline_image']['request_processors'][0]['ml_inference']['model_id']
        ds.store_in_dynamo('BEDROCK_Claude3_image_MODEL_ID', st.session_state.BEDROCK_Claude3_image_MODEL_ID )
            
else:
    st.session_state.llm_search_request_pipeline_image = "false"
ds.store_in_dynamo('llm_search_request_pipeline_image',st.session_state.llm_search_request_pipeline_image )



############################################################ 


    
if "REGION" not in st.session_state:
    st.session_state.REGION = ""
    
if "BEDROCK_MULTIMODAL_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_MULTIMODAL_MODEL_ID = ds.get_from_dynamo("BEDROCK_MULTIMODAL_MODEL_ID")
    
if "search_types" not in st.session_state:
    st.session_state.search_types =ds.get_from_dynamo("search_types")
    
if "KendraResourcePlanID" not in st.session_state:
    st.session_state.KendraResourcePlanID= ds.get_from_dynamo("KendraResourcePlanID")

if "SAGEMAKER_CrossEncoder_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID = ds.get_from_dynamo("SAGEMAKER_CrossEncoder_MODEL_ID")   
    
    
if "SAGEMAKER_SPARSE_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_SPARSE_MODEL_ID = ds.get_from_dynamo("SAGEMAKER_SPARSE_MODEL_ID")   
    
if "BEDROCK_TEXT_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_TEXT_MODEL_ID = ds.get_from_dynamo("BEDROCK_TEXT_MODEL_ID")  
    
if "BEDROCK_Claude3_image_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_Claude3_image_MODEL_ID = ds.get_from_dynamo("BEDROCK_Claude3_image_MODEL_ID")   
    
if "BEDROCK_Claude3_text_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_Claude3_text_MODEL_ID = ds.get_from_dynamo("BEDROCK_Claude3_text_MODEL_ID") 
#bytes_for_rekog = ""
bedrock_ = boto3.client('bedrock-runtime',region_name=st.session_state.REGION)
search_all_type = False
if(search_all_type==True):
    search_types = ['Keyword Search',
    'Vector Search', 
    'Multimodal Search',
    'NeuralSparse Search',
    ]
else:
    print("st.session_state.search_types")
    print(st.session_state.search_types) 
    search_types = (st.session_state.search_types).split(",")
    print(search_types)
from streamlit.components.v1 import html


def handle_input():
    if("text" in st.session_state.inputs_):
        if(st.session_state.inputs_["text"] != st.session_state.input_text):
            st.session_state.input_ndcg=0.0
    st.session_state.bytes_for_rekog = ""
    print("***")
    
    
    print("no image uploaded")
    st.session_state.input_imageUpload = 'no'
    st.session_state.input_image = ''


    inputs = {}
   
    for key in st.session_state:
        
        if(key.startswith('input_')):
            original_key = key.removeprefix('input_')
            inputs[original_key] = st.session_state[key]
            
    st.session_state.inputs_ = inputs
    
    #st.write(inputs) 
    question_with_id = {
        'question': inputs["text"],
        'id': len(st.session_state.questions)
    }
    st.session_state.questions = []
    st.session_state.questions.append(question_with_id)
    
    st.session_state.answers = []
    
    ans__ = all_search_execute.handler(inputs, st.session_state['session_id'])
    
    st.session_state.answers.append({
        'answer': ans__,#all_search_api.call(json.dumps(inputs), st.session_state['session_id']),
        'search_type':inputs['searchType'],
        'id': len(st.session_state.questions)
    })
    

    if(st.session_state.input_evaluate) == "enabled":
        llm_eval.eval(st.session_state.questions, st.session_state.answers)
    
def write_top_bar():

    col1, col2,col3,col4  = st.columns([2.5,35,8,7])
    with col1:
        st.image(TEXT_ICON, use_column_width='always')
    with col2:
        #st.markdown("")
        input = st.text_input( "Ask here",label_visibility = "collapsed",key="input_text",placeholder = "Type your query")
    with col3:
        play = st.button("Search",on_click=handle_input,key = "play")
        
    with col4:
        clear = st.button("Clear")
    
    
    return clear

clear = write_top_bar()

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
    
col1, col3, col4 = st.columns([70,28,2])

with col1:
    
    if(st.session_state.max_selections == "" or st.session_state.max_selections == "1"):
        st.session_state.max_selections = 1
    if(st.session_state.max_selections == "None"):
        st.session_state.max_selections = None
    search_type = st.multiselect('Select the Search type(s)',
    search_types,['Keyword Search'],
    max_selections = 1,
   
    key = 'input_searchType',
    help = "Select the type of Search"
    )

with col3:
    st.number_input("No. of docs", min_value=1, max_value=50, value=5, step=5,  key='input_K', help=None)

        

if(search_all_type == True or 1==1):
    with st.sidebar:
        st.page_link("/home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service/OpenSearchApp/app.py", label=":orange[Home]", icon="🏠")
       
        st.subheader(':blue[Filters]')
        def clear_filter():
            st.session_state.input_manual_filter="False"
            ds.store_in_dynamo('input_manual_filter',st.session_state.input_manual_filter )
            st.session_state.input_category=None
            st.session_state.input_gender=None
            st.session_state.input_price=(0,0)
            handle_input()
        filter_place_holder = st.container()
        with filter_place_holder:
            st.selectbox("Select one Category", ("accessories", "books","floral","furniture","hot_dispensed","jewelry","tools","apparel","cold_dispensed","food_service","groceries","housewares","outdoors","salty_snacks","videos","beauty","electronics","footwear","homedecor","instruments","seasonal"),index = None,key = "input_category")
            st.selectbox("Select one Gender", ("male","female"),index = None,key = "input_gender")
            st.slider("Select a range of price", 0, 2000, (0, 0),50, key = "input_price")

        if(st.session_state.input_category!=None or st.session_state.input_gender!=None or st.session_state.input_price!=(0,0)):
            st.session_state.input_manual_filter="True"
        else:
            st.session_state.input_manual_filter="False"

        ds.store_in_dynamo('input_manual_filter',st.session_state.input_manual_filter )

        clear_filter = st.button("Clear Filters",on_click=clear_filter)

        print("--------------------filters---------------")    
        print(st.session_state.input_gender)
        print(st.session_state.input_manual_filter)
        print("--------------------filters---------------") 



        ####### Filters   #########

        if('NeuralSparse Search' in st.session_state.search_types and st.session_state.neural_sparse_two_phase_search_pipeline != ''):
            st.subheader(':blue[Neural Sparse Search]')
            sparse_filter = st.slider('Prune ratio', 0.0, 1.0, 0.4,0.1,key = 'input_sparse_filter', help = 'A ratio that represents how to split the high-weight tokens and low-weight tokens. The threshold is the token’s maximum score multiplied by its prune_ratio. Valid range is [0,1]. Default is 0.4')


st.markdown('---')


def write_user_message(md,ans):
    #print(ans)
    ans = ans["answer"][0]
    col1, col2, col3 = st.columns([3,40,20])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
#         if(st.session_state.input_rewritten_query !=""):
#             display_query = st.session_state.input_rewritten_query['bool']["must"][0]["match"]["product_description"]["query"]
#         else:
#             display_query = md['question']
            
        st.markdown("<div style='fontSize:15px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'>Input Text: </div><div style='fontSize:25px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;font-style: italic;color:#e28743'>"+st.session_state.input_text+"</div>", unsafe_allow_html = True)#replace with md['question']
        if('query_sparse' in ans):
            with st.expander("Expanded Query:"):
                query_sparse = dict(sorted(ans['query_sparse'].items(), key=lambda item: item[1],reverse=True))
                filtered_query_sparse = dict()
                for key in query_sparse:
                    filtered_query_sparse[key] = round(query_sparse[key], 2)
                st.write(filtered_query_sparse)
        if(st.session_state.input_is_rewrite_query == 'enabled' and st.session_state.input_rewritten_query !=""):
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
                st.image(ans['image_url'])

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
                with st.expander("Document Metadata:",expanded = False):
                    # if("rekog" in ans):
                    #     div_size = [50,50]
                    # else:
                    #     div_size = [99,1]
                    # div1,div2 = st.columns(div_size)
                    # with div1:
                        
                    st.write(":green[default:]")
                    st.json({"category:":ans['category'],"price":str(ans['price']),"gender_affinity":ans['gender_affinity'],"style":ans['style']},expanded = True)
                    #with div2:
                    if("rekog" in ans):
                        st.write(":green[enriched:]")
                        st.json(ans['rekog'],expanded = True)
            with inner_col_1:
                
                if(st.session_state.input_evaluate == "enabled"):
                    with st.container(border = False):
                        if("relevant" in ans.keys()):
                            if(ans['relevant']==True):
                                st.write(":white_check_mark:")
                            else:
                                st.write(":x:")
                    
        i = i+1
  
            
            
    with col_3:
        if(index == len(st.session_state.questions)):

            rdn_key = ''.join([random.choice(string.ascii_letters)
                              for _ in range(10)])
            currentValue = "".join(st.session_state.input_searchType)+st.session_state.input_imageUpload+json.dumps(st.session_state.input_weightage)+st.session_state.input_NormType+st.session_state.input_CombineType+str(st.session_state.input_K)+st.session_state.input_sparse+st.session_state.input_reranker+st.session_state.input_is_rewrite_query+st.session_state.input_evaluate+st.session_state.input_image+st.session_state.input_rad_1+st.session_state.input_reranker+st.session_state.input_hybridType+st.session_state.input_manual_filter
            oldValue = "".join(st.session_state.inputs_["searchType"])+st.session_state.inputs_["imageUpload"]+str(st.session_state.inputs_["weightage"])+st.session_state.inputs_["NormType"]+st.session_state.inputs_["CombineType"]+str(st.session_state.inputs_["K"])+st.session_state.inputs_["sparse"]+st.session_state.inputs_["reranker"]+st.session_state.inputs_["is_rewrite_query"]+st.session_state.inputs_["evaluate"]+st.session_state.inputs_["image"]+st.session_state.inputs_["rad_1"]+st.session_state.inputs_["reranker"]+st.session_state.inputs_["hybridType"]+st.session_state.inputs_["manual_filter"]
            
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
