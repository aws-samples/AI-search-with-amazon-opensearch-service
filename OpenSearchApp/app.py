import streamlit as st
from PIL import Image
import base64
import yaml
from yaml.loader import SafeLoader
from streamlit_javascript import st_javascript
import streamlit_authenticator as stauth
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth
import requests
import json

st.set_page_config(
    
    #page_title="Semantic Search using OpenSearch",
    layout="wide",
    page_icon="/home/ubuntu/images/opensearch_mark_default.png"
)

if "play_disabled" not in st.session_state:
    st.session_state.play_disabled = True

#DOMAIN_ENDPOINT =   "search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com" #"search-opensearchservi-rimlzstyyeih-3zru5p2nxizobaym45e5inuayq.us-west-2.es.amazonaws.com" 
REGION = "us-east-1" #'us-west-2'#
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, 'es', session_token=credentials.token)

def generate_images():
    
    # 1. get the cfn outputs
    
    cfn = boto3.client('cloudformation',region_name='us-east-1')

    # response = cfn.list_stacks(StackStatusFilter=['CREATE_COMPLETE','UPDATE_COMPLETE'])

    # for cfns in response['StackSummaries']:
    #     if('TemplateDescription' in cfns.keys()):
    #         if('ml search' in cfns['TemplateDescription']):
    #             stackname = cfns['StackName']
    

    # response = cfn.describe_stack_resources(
    #     StackName=stackname
    # )
    

    # cfn_outputs = cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']

    # for output in cfn_outputs:
    #     if('OpenSearchDomainEndpoint' in output['OutputKey']):
    #         OpenSearchDomainEndpoint = output['OutputValue']
            
    #     if('EmbeddingEndpointName' in output['OutputKey']):
    #         SagemakerEmbeddingEndpoint = output['OutputValue']
            
    #     if('s3' in output['OutputKey'].lower()):
    #         s3_bucket = output['OutputValue']
            

    # region = boto3.Session().region_name  
            

    # account_id = boto3.client('sts').get_caller_identity().get('Account')



    # print("stackname: "+stackname)
    # print("account_id: "+account_id)  
    # print("region: "+region)
    # print("SagemakerEmbeddingEndpoint: "+SagemakerEmbeddingEndpoint)
    # print("OpenSearchDomainEndpoint: "+OpenSearchDomainEndpoint)
    # print("S3 Bucket: "+s3_bucket)
    
    # 2. Create the OpenSearch-Sagemaker ML connector
    
    
    # 3. Register and deploy the model
    
    REGION = "us-east-1" 
    SAGEMAKER_MODEL_ID = 'uPQAE40BnrP7K1qW-Alx' 
    BEDROCK_TEXT_MODEL_ID = 'iUvGQYwBuQkLO8mDfE0l'
    BEDROCK_MULTIMODAL_MODEL_ID = 'o6GykYwB08ySW4fgXdf3'
    SAGEMAKER_SPARSE_MODEL_ID = 'srrJ-owBQhe1aB-khx2n'
    
    
    # 4. Create ingest pipelines
    
    pipeline = "ml-ingest-pipeline"
    
    # 5. Create index
    
    ml_index = "retail-ml-search-index"
    
    # 6. Index data
    
    
    # 7. Enable Playground
    
    st.session_state.play_disabled = False
    
    
    
    
                
    #st.switch_page('pages/Semantic_Search.py')



headers = {"Content-Type": "application/json"}

input_host = st.text_input( "OpenSearch domain URL",key="input_host",placeholder = "Opensearch host",value = "https://search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com/")
input_index = st.text_input( "OpenSearch domain index",key="input_index",placeholder = "Opensearch index name", value = "raw-retail-ml-search-index")
url = input_host + input_index
get_fileds = st.button('Get field metadata')
playground = st.button('Launch Playground', disabled = st.session_state.play_disabled)
if(playground):
    st.switch_page('pages/Semantic_Search.py')
if(get_fileds):
    
    r = requests.get(url, auth=awsauth,  headers=headers)
    mappings= json.loads(r.text)[input_index]["mappings"]["properties"]
    fields = []
    for i in mappings.keys():
        if(mappings[i]["type"] != 'knn_vector' and mappings[i]["type"] != "rank_features"):
            fields.append({i:mappings[i]["type"]})
    col1,col2 = st.columns([50,50])
    with col1:
        st.write(fields)
    with col2:
        input_vectors = st.text_input( "comma separated field names that needs to be vectorised",key="input_vectors",placeholder = "field1,field2")
        submit = st.button("Submit",on_click = generate_images)
        
        
    