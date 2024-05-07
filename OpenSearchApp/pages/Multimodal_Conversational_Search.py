import streamlit as st
import uuid
import os
import sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/semantic_search")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/RAG")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/utilities")
import boto3
import requests
from boto3 import Session
import botocore.session
import json
import random
import string
import rag_DocumentLoader 
import rag_DocumentSearcher
import pandas as pd
from PIL import Image 
import shutil
import base64
import time
import botocore
from langchain.callbacks.base import BaseCallbackHandler
import streamlit_nested_layout
from IPython.display import clear_output, display, display_markdown, Markdown
from requests_aws4auth import AWS4Auth
from requests.auth import HTTPBasicAuth



st.set_page_config(
    #page_title="Semantic Search using OpenSearch",
    layout="wide",
    page_icon="images/opensearch_mark_default.png"
)
parent_dirname = "/".join((os.path.dirname(__file__)).split("/")[0:-1])
USER_ICON = "images/user.png"
AI_ICON = "images/opensearch-twitter-card.png"
REGENERATE_ICON = "images/regenerate.png"
s3_bucket_ = "pdf-repo-uploads"
            #"pdf-repo-uploads"
polly_client = boto3.Session(
            region_name='us-east-1').client('polly')

# Check if the user ID is already stored in the session state
if 'user_id' in st.session_state:
    user_id = st.session_state['user_id']
    #print(f"User ID: {user_id}")

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

if "questions_" not in st.session_state:
    st.session_state.questions_ = []

if "answers_" not in st.session_state:
    st.session_state.answers_ = []

if "input_index" not in st.session_state:
    st.session_state.input_index = "globalwarmingnew"#"hpijan2024hometrack_no_img_no_table"
    
if "input_is_rerank" not in st.session_state:
    st.session_state.input_is_rerank = True
    
    
if "input_table_with_sql" not in st.session_state:
    st.session_state.input_table_with_sql = False
    
if "input_query" not in st.session_state:
    st.session_state.input_query="What is the projected energy percentage from renewable sources in future?"#"Which city in United Kingdom has the highest average housing price ?"#"How many aged above 85 years died due to covid ?"# What is the projected energy from renewable sources ?"


if "input_rag_searchType" not in st.session_state:
    st.session_state.input_rag_searchType = ["Sparse Search"]
    


        
region = 'us-east-1'
bedrock_runtime_client = boto3.client('bedrock-runtime',region_name=region)
output = []
service = 'es'

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

################ OpenSearch Py client #####################
    
# credentials = boto3.Session().get_credentials()
# awsauth = AWSV4SignerAuth(credentials, region, service)

# ospy_client = OpenSearch(
#     hosts = [{'host': 'search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com', 'port': 443}],
#     http_auth = awsauth,
#     use_ssl = True,
#     verify_certs = True,
#     connection_class = RequestsHttpConnection,
#     pool_maxsize = 20
# )

################# using boto3 credentials ###################


credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
service = 'es'


################# using boto3 credentials ####################



# if "input_searchType" not in st.session_state:
#     st.session_state.input_searchType = "Conversational Search (RAG)"

# if "input_temperature" not in st.session_state:
#     st.session_state.input_temperature = "0.001"

# if "input_topK" not in st.session_state:
#     st.session_state.input_topK = 200

# if "input_topP" not in st.session_state:
#     st.session_state.input_topP = 0.95

# if "input_maxTokens" not in st.session_state:
#     st.session_state.input_maxTokens = 1024


def write_logo():
    col1, col2, col3 = st.columns([5, 1, 5])
    with col2:
        st.image(AI_ICON, use_column_width='always') 

def write_top_bar():
    col1, col2 = st.columns([77,23])
    with col1:
        st.write("")
        st.header("Chat with your data",divider='rainbow')
        
        #st.image(AI_ICON, use_column_width='always')
    
    with col2:
        st.write("")
        st.write("")
        clear = st.button("CLEAR")
    st.write("")
    st.write("")
    return clear

clear = write_top_bar()

if clear:
    st.session_state.questions_ = []
    st.session_state.answers_ = []
    st.session_state.input_query=""
    # st.session_state.input_searchType="Conversational Search (RAG)"
    # st.session_state.input_temperature = "0.001"
    # st.session_state.input_topK = 200
    # st.session_state.input_topP = 0.95
    # st.session_state.input_maxTokens = 1024


def handle_input():
    print("Question: "+st.session_state.input_query)
    print("-----------")
    print("\n\n")
    if(st.session_state.input_query==''):
        return ""
    inputs = {}
    for key in st.session_state:
        if key.startswith('input_'):
            inputs[key.removeprefix('input_')] = st.session_state[key]
    st.session_state.inputs_ = inputs
    
    #######
    
    
    #st.write(inputs) 
    question_with_id = {
        'question': inputs["query"],
        'id': len(st.session_state.questions_)
    }
    st.session_state.questions_.append(question_with_id)
    out_ = rag_DocumentSearcher.query_(awsauth, inputs, st.session_state['session_id'],st.session_state.input_rag_searchType)
    st.session_state.answers_.append({
        'answer': out_['text'],
        'source':out_['source'],
        'id': len(st.session_state.questions_),
        'image': out_['image'],
        'table':out_['table']
    })
    st.session_state.input_query=""
    

    
# search_type = st.selectbox('Select the Search type',
#     ('Conversational Search (RAG)',
#     'OpenSearch vector search', 
#     'LLM Text Generation'
#     ),
   
#     key = 'input_searchType',
#     help = "Select the type of retriever\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers_)"
#     )

# col1, col2, col3, col4 = st.columns(4)
    
# with col1:
#     st.text_input('Temperature', value = "0.001", placeholder='LLM Temperature', key = 'input_temperature',help = "Set the temperature of the Large Language model. \n Note: 1. Set this to values lower to 1 in the order of 0.001, 0.0001, such low values reduces hallucination and creativity in the LLM response; 2. This applies only when LLM is a part of the retriever pipeline")
# with col2:
#     st.number_input('Top K', value = 200, placeholder='Top K', key = 'input_topK', step = 50, help = "This limits the LLM's predictions to the top k most probable tokens at each step of generation, this applies only when LLM is a prt of the retriever pipeline")
# with col3:
#     st.number_input('Top P', value = 0.95, placeholder='Top P', key = 'input_topP', step = 0.05, help = "This sets a threshold probability and selects the top tokens whose cumulative probability exceeds the threshold while the tokens are generated by the LLM")
# with col4:
#     st.number_input('Max Output Tokens', value = 500, placeholder='Max Output Tokens', key = 'input_maxTokens', step = 100, help = "This decides the total number of tokens generated as the final response. Note: Values greater than 1000 takes longer response time")

# st.markdown('---')


def write_user_message(md):
    col1, col2 = st.columns([3,97])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        #st.warning(md['question'])

        st.markdown("<div style='color:#e28743';font-size:18px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;font-style: italic;'>"+md['question']+"</div>", unsafe_allow_html = True)
       


def render_answer(question,answer,index,res_img):
    
    
    col1, col2, col_3 = st.columns([4,74,22])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        ans_ = answer['answer']
        st.write(ans_)
       
        
        
        # def stream_():
        #     #use for streaming response on the client side
        #     for word in ans_.split(" "):
        #         yield word + " "
        #         time.sleep(0.04)
        #     #use for streaming response from Llm directly
        #     if(isinstance(ans_,botocore.eventstream.EventStream)):
        #         for event in ans_:
        #             chunk = event.get('chunk')
                    
        #             if chunk:
                        
        #                 chunk_obj = json.loads(chunk.get('bytes').decode())
                        
        #                 if('content_block' in chunk_obj or ('delta' in chunk_obj and 'text' in chunk_obj['delta'])):
        #                     key_ = list(chunk_obj.keys())[2]
        #                     text = chunk_obj[key_]['text']
                            
        #                     clear_output(wait=True)
        #                     output.append(text)
        #                     yield text
        #                     time.sleep(0.04)
            
                
        
        # if(index == len(st.session_state.questions_)):
        #     st.write_stream(stream_)
        #     if(isinstance(st.session_state.answers_[index-1]['answer'],botocore.eventstream.EventStream)):
        #         st.session_state.answers_[index-1]['answer'] = "".join(output)
        # else:
        #     st.write(ans_)
        

        polly_response = polly_client.synthesize_speech(VoiceId='Joanna',
                        OutputFormat='ogg_vorbis', 
                        Text = ans_,
                        Engine = 'neural')

        audio_col1, audio_col2 = st.columns([50,50])
        with audio_col1:
            st.audio(polly_response['AudioStream'].read(), format="audio/ogg")
                
        
        
        #st.markdown("<div style='font-size:18px;padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;border-radius: 10px;'>"+ans_+"</div>", unsafe_allow_html = True)
    #st.markdown("<div style='color:#e28743';padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'><b>Relevant images from the document :</b></div>", unsafe_allow_html = True)
    #st.write("")
    colu1,colu2,colu3 = st.columns([4,82,20])
    with colu2:
        with st.expander("Relevant Sources:"):
            with st.container():
                if(len(res_img)>0):
                    with st.expander("Images:"):
                        col3,col4,col5 = st.columns([33,33,33])
                        cols = [col3,col4]
                        idx = 0
                        #print(res_img)
                        for img_ in res_img:
                            if(img_['file'].lower()!='none' and idx < 2):
                                img = img_['file'].split(".")[0]
                                caption = img_['caption']
                                
                                with cols[idx]:
                                    
                                    st.image(parent_dirname+"/figures/"+st.session_state.input_index+"/"+img+".jpg")
                                    #st.write(caption)
                                idx = idx+1
                #st.markdown("<div style='color:#e28743';padding:3px 7px 3px 7px;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 10px;'><b>Sources from the document:</b></div>", unsafe_allow_html = True)
                if(len(answer["table"] )>0):
                    with st.expander("Table:"):
                        df = pd.read_csv(answer["table"][0]['name'],skipinitialspace = True, on_bad_lines='skip',delimiter='`')
                        df.fillna(method='pad', inplace=True)
                        st.table(df)
                with st.expander("Raw sources:"):
                    st.write(answer["source"])
                
        
        
    with col_3:
        
        #st.markdown("<div style='color:#e28743;borderWidth: 0px;borderColor: red;borderStyle: solid;width: fit-content;height: fit-content;border-radius: 5px;'><b>"+",".join(st.session_state.input_rag_searchType)+"</b></div>", unsafe_allow_html = True)
        
       
        
        if(index == len(st.session_state.questions_)):

            rdn_key = ''.join([random.choice(string.ascii_letters)
                              for _ in range(10)])
            currentValue = ''.join(st.session_state.input_rag_searchType)+str(st.session_state.input_is_rerank)+str(st.session_state.input_table_with_sql)+st.session_state.input_index
            oldValue = ''.join(st.session_state.inputs_["rag_searchType"])+str(st.session_state.inputs_["is_rerank"])+str(st.session_state.inputs_["table_with_sql"])+str(st.session_state.inputs_["index"])
            #print("changing values-----------------")
            def on_button_click():
                # print("button clicked---------------")
                # print(currentValue)
                # print(oldValue)
                if(currentValue!=oldValue or 1==1): 
                    #print("----------regenerate----------------")
                    st.session_state.input_query = st.session_state.questions_[-1]["question"]
                    st.session_state.answers_.pop()
                    st.session_state.questions_.pop()
                    
                    handle_input()
                    with placeholder.container():
                        render_all()

            if("currentValue"  in st.session_state):
                del st.session_state["currentValue"]

            try:
                del regenerate
            except:
                pass  

            #print("------------------------")
            #print(st.session_state)

            placeholder__ = st.empty()
            
            placeholder__.button("🔄",key=rdn_key,on_click=on_button_click)
     
#Each answer will have context of the question asked in order to associate the provided feedback with the respective question
def write_chat_message(md, q,index):
    res_img = md['image']
    #st.session_state['session_id'] = res['session_id']   to be added in memory
    chat = st.container()
    with chat:
        #print("st.session_state.input_index------------------")
        #print(st.session_state.input_index)
        render_answer(q,md,index,res_img)
    
def render_all():  
    index = 0
    for (q, a) in zip(st.session_state.questions_, st.session_state.answers_):
        index = index +1
        
        write_user_message(q)
        write_chat_message(a, q,index)

placeholder = st.empty()
with placeholder.container():
  render_all()

st.markdown("")
col_2, col_3 = st.columns([75,20])
#col_1, col_2, col_3 = st.columns([7.5,71.5,22])
# with col_1:
#     st.markdown("<p style='padding:0px 0px 0px 0px; color:#FF9900;font-size:120%'><b>Ask:</b></p>",unsafe_allow_html=True, help = 'Enter the questions and click on "GO"')
    
with col_2:
    #st.markdown("")
    input = st.text_input( "Ask here",label_visibility = "collapsed",key="input_query")
with col_3:
    #hidden = st.button("RUN",disabled=True,key = "hidden")
    play = st.button("GO",on_click=handle_input,key = "play")
with st.sidebar:
    st.page_link("/home/ubuntu/AI-search-with-amazon-opensearch-service/OpenSearchApp/app.py", label=":orange[Home]", icon="🏠")
    st.subheader(":blue[Sample Data]")
    coln_1,coln_2 = st.columns([70,30])
    # index_select = st.radio("Choose one index",["UK Housing","Covid19 impacts on Ireland","Environmental Global Warming","BEIR Research"],
    #                         captions = ['[preview](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/HPI-Jan-2024-Hometrack.pdf)',
    #                                     '[preview](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/covid19_ie.pdf)',
    #                                     '[preview](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/global_warming.pdf)',
    #                                     '[preview](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/BEIR.pdf)'],
    #                         key="input_rad_index")
    with coln_1:
        index_select = st.radio("Choose one index",["Global Warming stats","UK Housing","Covid19 impacts on Ireland"],key="input_rad_index")
    with coln_2:
        st.markdown("<p style='font-size:15px'>Preview file</p>",unsafe_allow_html=True)
        st.write("[:eyes:](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/HPI-Jan-2024-Hometrack.pdf)")
        st.write("[:eyes:](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/covid19_ie.pdf)")
        st.write("[:eyes:](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/global_warming.pdf)")
        #st.write("[:eyes:](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/b559f82c07dfcca973f457c0a15d6444752553ab/rag/sample_pdfs/BEIR.pdf)")
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
    # Initialize boto3 to use the S3 client.
    s3_client = boto3.resource('s3')
    bucket=s3_client.Bucket(s3_bucket_)

    objects = bucket.objects.filter(Prefix="sample_pdfs/")
    urls = []

    client = boto3.client('s3')

    for obj in objects:
        if obj.key.endswith('.pdf'): 

            # Generate the S3 presigned URL
            s3_presigned_url = client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': s3_bucket_,
                    'Key': obj.key
                },
                ExpiresIn=3600
            )

            # Print the created S3 presigned URL
            print(s3_presigned_url)
            urls.append(s3_presigned_url)
            #st.write("["+obj.key.split('/')[1]+"]("+s3_presigned_url+")")
            st.link_button(obj.key.split('/')[1], s3_presigned_url)
    
    st.subheader(":blue[Your multi-modal documents]")
    pdf_doc_ = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=False)
                    
                
    pdf_docs = [pdf_doc_]
    if st.button("Process"):
        with st.spinner("Processing"):
            if os.path.isdir(parent_dirname+"/pdfs"):
                shutil.rmtree(parent_dirname+"/pdfs")

            os.mkdir(parent_dirname+"/pdfs")
            for pdf_doc in pdf_docs:
                print(type(pdf_doc))
                pdf_doc_name = (pdf_doc.name).replace(" ","_")
                with open(os.path.join(parent_dirname+"/pdfs",pdf_doc_name),"wb") as f: 
                    f.write(pdf_doc.getbuffer())  
                    
                request_ = { "bucket": s3_bucket_,"key": pdf_doc_name}
                rag_DocumentLoader.load_docs(request_)
                print('lambda done')
        st.success('you can start searching on your PDF')
        
    ############## haystach demo temporary addition ############    
    # st.subheader(":blue[Multimodality]")
    # colu1,colu2 = st.columns([50,50])
    # with colu1:
    #     in_images = st.toggle('Images', key = 'in_images', disabled = False)
    # with colu2:
    #     in_tables = st.toggle('Tables', key = 'in_tables', disabled = False)   
    # if(in_tables):
    #     st.session_state.input_table_with_sql = True
    # else:
    #     st.session_state.input_table_with_sql = False
        
     ############## haystach demo temporary addition ############       
    if(pdf_doc_ is None or pdf_doc_ == ""):
        if(index_select == "Global Warming stats"):
            st.session_state.input_index = "globalwarmingnew"
        if(index_select == "Covid19 impacts on Ireland"):
            st.session_state.input_index = "covid19ie"#"choosetheknnalgorithmforyourbillionscaleusecasewithopensearchawsbigdatablog"
        if(index_select == "BEIR"):
            st.session_state.input_index = "2104"
        if(index_select == "UK Housing"):
            st.session_state.input_index = "hpijan2024hometrack"
            # if(in_images == True and in_tables == True):
            #     st.session_state.input_index = "hpijan2024hometrack"
            # else:
            #     if(in_images == True and in_tables == False):
            #         st.session_state.input_index = "hpijan2024hometrackno_table"
            #     else:
            #         if(in_images == False and in_tables == True):
            #             st.session_state.input_index = "hpijan2024hometrackno_images"
            #         else:   
            #             st.session_state.input_index = "hpijan2024hometrack_no_img_no_table"
                
                    
    # if(in_images):
    #     st.session_state.input_include_images = True
    # else:
    #     st.session_state.input_include_images = False
    # if(in_tables):
    #     st.session_state.input_include_tables = True
    # else:
    #     st.session_state.input_include_tables = False
    
    
    
    st.subheader(":blue[Retriever]")
    search_type = st.multiselect('Select the Retriever(s)',
    ['Keyword Search',
    'Vector Search', 
    'Sparse Search',
    ],
    ['Sparse Search'],

    key = 'input_rag_searchType',
    help = "Select the type of Search, adding more than one search type will activate hybrid search"#\n1. Conversational Search (Recommended) - This will include both the OpenSearch and LLM in the retrieval pipeline \n (note: This will put opensearch response as context to LLM to answer) \n2. OpenSearch vector search - This will put only OpenSearch's vector search in the pipeline, \n(Warning: this will lead to unformatted results )\n3. LLM Text Generation - This will include only LLM in the pipeline, \n(Warning: This will give hallucinated and out of context answers)"
    )
    
    re_rank = st.checkbox('Re-rank results', key = 'input_re_rank', disabled = False, value = True, help = "Checking this box will re-rank the results using a cross-encoder model")
        
    if(re_rank):
        st.session_state.input_is_rerank = True
    else:
        st.session_state.input_is_rerank = False
        
        
        
