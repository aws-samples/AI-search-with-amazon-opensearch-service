import streamlit as st
from PIL import Image
import base64
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth
from requests.auth import HTTPBasicAuth
import requests
import json
import time
import os
import urllib.request
import tarfile
import subprocess
from ruamel.yaml import YAML
from PIL import Image
import base64
import sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-1])+"/semantic_search")
import dynamo_state as ds

st.set_page_config(
    
    #page_title="Semantic Search using OpenSearch",
    layout="wide",
    page_icon="/home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service/OpenSearchApp/images/opensearch_logo.png"
)

col_0_1,col_0_2,col_0_3= st.columns([10,90,30])
with col_0_1:
    st.image("/home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service/OpenSearchApp/images/opensearch_logo.png", use_column_width='always')
with col_0_2:
    st.header("Intelligent search with Amazon OpenSearch Service")
st.write("")
st.write("")
st.write("")
st.session_state.REGION = ds.get_region()


if "play_disabled" not in st.session_state:
    st.session_state.play_disabled = ds.get_from_dynamo("play_disabled")
    
    
    
if "index_map" not in st.session_state:
    st.session_state.index_map = {}
    
if "OpenSearchDomainEndpoint" not in st.session_state:
    st.session_state.OpenSearchDomainEndpoint = ds.get_from_dynamo("OpenSearchDomainEndpoint")
    
if "KendraResourcePlanID" not in st.session_state:
    st.session_state.KendraResourcePlanID = ds.get_from_dynamo("KendraResourcePlanID")
    
if "REGION" not in st.session_state:
    st.session_state.REGION = ""
   
if "WebappRoleArn" not in st.session_state:
    st.session_state.WebappRoleArn = ds.get_from_dynamo("WebappRoleArn")
    
if "BEDROCK_MULTIMODAL_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_MULTIMODAL_MODEL_ID = ds.get_from_dynamo("BEDROCK_MULTIMODAL_MODEL_ID")
    
if "BEDROCK_MULTIMODAL_CONNECTOR_ID" not in st.session_state:
    st.session_state.BEDROCK_MULTIMODAL_CONNECTOR_ID = ds.get_from_dynamo("BEDROCK_MULTIMODAL_CONNECTOR_ID")
    
if "max_selections" not in st.session_state:
    st.session_state.max_selections = ds.get_from_dynamo("max_selections")
    
if "sagemaker_re_ranker" not in st.session_state:
    st.session_state.sagemaker_re_ranker = ds.get_from_dynamo("sagemaker_re_ranker")
    
if "bedrock_re_ranker" not in st.session_state:
    st.session_state.bedrock_re_ranker = ds.get_from_dynamo("bedrock_re_ranker")
    
if "llm_search_request_pipeline" not in st.session_state:
    st.session_state.llm_search_request_pipeline = ds.get_from_dynamo("llm_search_request_pipeline")
    
if "neural_sparse_two_phase_search_pipeline" not in st.session_state:
    st.session_state.neural_sparse_two_phase_search_pipeline = ds.get_from_dynamo("neural_sparse_two_phase_search_pipeline")   
        
if "search_types" not in st.session_state:
    st.session_state.search_types =ds.get_from_dynamo("search_types")
    if(st.session_state.search_types == ""):
        st.session_state.search_types = "Keyword Search"

if "SAGEMAKER_SPARSE_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_SPARSE_MODEL_ID = ds.get_from_dynamo("SAGEMAKER_SPARSE_MODEL_ID")   
    
if "SAGEMAKER_SPARSE_CONNECTOR_ID" not in st.session_state:
    st.session_state.SAGEMAKER_SPARSE_CONNECTOR_ID = ds.get_from_dynamo("SAGEMAKER_SPARSE_CONNECTOR_ID")   

if "BEDROCK_Rerank_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_Rerank_MODEL_ID = ds.get_from_dynamo("BEDROCK_Rerank_MODEL_ID")   
    
if "BEDROCK_Rerank_CONNECTOR_ID" not in st.session_state:
    st.session_state.BEDROCK_Rerank_CONNECTOR_ID = ds.get_from_dynamo("BEDROCK_Rerank_CONNECTOR_ID") 
    
if "SAGEMAKER_CrossEncoder_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID = ds.get_from_dynamo("SAGEMAKER_CrossEncoder_MODEL_ID")   
    
if "SAGEMAKER_CrossEncoder_CONNECTOR_ID" not in st.session_state:
    st.session_state.SAGEMAKER_CrossEncoder_CONNECTOR_ID = ds.get_from_dynamo("SAGEMAKER_CrossEncoder_CONNECTOR_ID")  
    
if "BEDROCK_TEXT_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_TEXT_MODEL_ID = ds.get_from_dynamo("BEDROCK_TEXT_MODEL_ID")  
    
if "BEDROCK_TEXT_CONNECTOR_ID" not in st.session_state:
    st.session_state.BEDROCK_TEXT_CONNECTOR_ID = ds.get_from_dynamo("BEDROCK_TEXT_CONNECTOR_ID")  
    
if "BEDROCK_Claude3_image_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_Claude3_image_MODEL_ID = ds.get_from_dynamo("BEDROCK_Claude3_image_MODEL_ID")   
    
if "BEDROCK_Claude3_image_CONNECTOR_ID" not in st.session_state:
    st.session_state.BEDROCK_Claude3_image_CONNECTOR_ID = ds.get_from_dynamo("BEDROCK_Claude3_image_CONNECTOR_ID")  
    
if "BEDROCK_Claude3_text_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_Claude3_text_MODEL_ID = ds.get_from_dynamo("BEDROCK_Claude3_text_MODEL_ID")   
    
if "BEDROCK_Claude3_text_CONNECTOR_ID" not in st.session_state:
    st.session_state.BEDROCK_Claude3_text_CONNECTOR_ID = ds.get_from_dynamo("BEDROCK_Claude3_text_CONNECTOR_ID")  
    

isExist = os.path.exists("/home/ec2-user/SageMaker/images_retail")
if not isExist:   
    os.makedirs('/home/ec2-user/SageMaker/images_retail')
    metadata_file = urllib.request.urlretrieve('https://aws-blogs-artifacts-public.s3.amazonaws.com/BDB-3144/products-data.yml', '/home/ec2-user/SageMaker/images_retail/products.yaml')
    img_filename,headers= urllib.request.urlretrieve('https://aws-blogs-artifacts-public.s3.amazonaws.com/BDB-3144/images.tar.gz', '/home/ec2-user/SageMaker/images_retail/images.tar.gz')              
    print(img_filename)
    file = tarfile.open('/home/ec2-user/SageMaker/images_retail/images.tar.gz')
    file.extractall('/home/ec2-user/SageMaker/images_retail')
    file.close()
    #remove images.tar.gz
    os.remove('/home/ec2-user/SageMaker/images_retail/images.tar.gz')
    
preview_data = ["footwear","jewelry","apparel","beauty","housewares"]
yaml = YAML()
preview_contain = st.container()
preview_items = yaml.load(open('/home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service/preview_data.yaml'))

with st.expander("Preview retail data samples",expanded = False):
    samp1, samp2,samp3,samp4  = st.columns([25,25,25,25])
    col_array = [samp1, samp2,samp3,samp4]
    count = 0
    for item in preview_items:

        count = count + 1
        fileshort = "/home/ec2-user/SageMaker/images_retail/"+item["category"]+"/"+item["image"]

        payload = {}
        payload['product_description'] = item['description']
        payload['caption'] = item['name']
        payload['category'] = item['category']
        payload['price'] = item['price']
        if('gender_affinity' in item):
            if(item['gender_affinity'] == 'M'):
                payload['gender_affinity'] = 'Male'
            else:
                if(item['gender_affinity'] == 'F'):
                    payload['gender_affinity'] = 'Female'
                else:
                    payload['gender_affinity'] = item['gender_affinity']
        if('style' in item):          
            payload['style'] = item['style']
        with col_array[count-1]:
            if(count == 1):
                st.subheader(item['category'])
            else:
                st.subheader("")

            st.image(fileshort,use_column_width="always")
            st.write(":orange["+payload['caption']+"]")
            st.json(payload,expanded = False)
        if(count == 4):
            count = 0

cfn = boto3.client('cloudformation',region_name=st.session_state.REGION)

response = cfn.list_stacks(StackStatusFilter=['CREATE_COMPLETE','UPDATE_COMPLETE'])

for cfns in response['StackSummaries']:
    if('TemplateDescription' in cfns.keys()):
        if('NextGen ML search' in cfns['TemplateDescription']):
            stackname = cfns['StackName']


response = cfn.describe_stack_resources(
    StackName=stackname
)


cfn_outputs = cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']

for output in cfn_outputs:
    if('OpenSearchDomainEndpoint' in output['OutputKey']):
        OpenSearchDomainEndpoint = output['OutputValue']
        
    if('WebappRoleArn' in output['OutputKey']):
        WebappRoleArn = output['OutputValue']
        
    if('KendraResourcePlanID' in output['OutputKey']):
        KendraResourcePlanID = output['OutputValue']
    else:
        KendraResourcePlanID = ""
    
        
    
        
        
        


        

account_id = boto3.client('sts').get_caller_identity().get('Account')





print("stackname: "+stackname)
print("account_id: "+account_id)  
#print("region: "+region)
print("OpenSearchDomainEndpoint: "+OpenSearchDomainEndpoint)
print("WebappRoleArn: "+WebappRoleArn)

st.session_state.OpenSearchDomainEndpoint = OpenSearchDomainEndpoint
st.session_state.KendraResourcePlanID = KendraResourcePlanID
ds.store_in_dynamo('KendraResourcePlanID',st.session_state.KendraResourcePlanID )
ds.store_in_dynamo('OpenSearchDomainEndpoint',st.session_state.OpenSearchDomainEndpoint )
st.session_state.WebappRoleArn = WebappRoleArn
ds.store_in_dynamo('WebappRoleArn',st.session_state.WebappRoleArn )
ds.store_in_dynamo('REGION',st.session_state.REGION )


host = 'https://'+OpenSearchDomainEndpoint+'/'
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, st.session_state.REGION, service, session_token=credentials.token)
headers = {"Content-Type": "application/json"}
dash_url = 'https://'+OpenSearchDomainEndpoint+'/'+ '_dashboards/app/security-dashboards-plugin#/roles/edit/all_access/mapuser'

exists_ = requests.head(host+'demostore-search-index', auth=awsauth,headers=headers)
print(exists_)

if('403' in str(exists_)):
    st.error('Please add the IAM role, "'+st.session_state.WebappRoleArn+'" to OpenSearch backend roles [here](%s)' % dash_url ,icon = "🚨")
    st.stop()
if('404' in str(exists_) or '403' in str(exists_)):
    
    st.session_state.play_disabled = 'True'
else:
    st.session_state.play_disabled = 'False'
ds.store_in_dynamo('play_disabled',st.session_state.play_disabled)
    


opensearch_sparse_search_pipeline = (requests.get(host+'_search/pipeline/neural_sparse_two_phase_search_pipeline', auth=awsauth,headers=headers)).text

if(opensearch_sparse_search_pipeline!='{}'):
    st.session_state.neural_sparse_two_phase_search_pipeline = opensearch_sparse_search_pipeline
else:
    st.session_state.neural_sparse_two_phase_search_pipeline = ""
        
ds.store_in_dynamo('neural_sparse_two_phase_search_pipeline',st.session_state.neural_sparse_two_phase_search_pipeline )



###### check for search pipelines #######

def create_ml_connectors():
    
    permissions = {
    "persistent": {
        "plugins.ml_commons.trusted_connector_endpoints_regex": [
          "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
          "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
            ]
        }
    }

    permission_res = requests.put(host+'/_cluster/settings',json = permissions, auth=awsauth,headers=headers)
    
    


    remote_ml = {
                "SAGEMAKER_SPARSE":
                 {
                     "endpoint_url":"https://runtime.sagemaker."+st.session_state.REGION+".amazonaws.com/endpoints/neural-sparse-model/invocations",
                     "pre_process_fun": '\n    StringBuilder builder = new StringBuilder();\n    builder.append("\\"");\n    builder.append(params.text_docs[0]);\n    builder.append("\\"");\n    def parameters = "{" +"\\"inputs\\":" + builder + "}";\n    return "{" +"\\"parameters\\":" + parameters + "}";\n    ', 
                   #"post_process_fun": '\n    def name = "sentence_embedding";\n    def dataType = "FLOAT32";\n    if (params.result == null || params.result.length == 0) {\n        return null;\n    }\n    def shape = [params.result[0].length];\n    def json = "{" +\n               "\\"name\\":\\"" + name + "\\"," +\n               "\\"data_type\\":\\"" + dataType + "\\"," +\n               "\\"shape\\":" + shape + "," +\n               "\\"data\\":" + params.result[0] +\n               "}";\n    return json;\n    ',
                    "request_body": """["${parameters.inputs}"]"""
             
                 }
            }

    connector_path_url = host+'_plugins/_ml/connectors/_create'
    

    for remote_ml_key in remote_ml.keys():
        name = remote_ml_key+": EMBEDDING"
            
        #create connector
        payload_1 = {
        "name": name, 
        "description": "Connector for "+remote_ml_key+" remote model",
        "version": 1,
        "protocol": "aws_sigv4",
        "credential": {
            "roleArn": "arn:aws:iam::"+account_id+":role/opensearch-sagemaker-role"
        },
        "parameters": {
            "region": st.session_state.REGION,
            "service_name": (remote_ml_key.split("_")[0]).lower(),
            "input_docs_processed_step_size": "2"
        },
        "actions": [
            {
                "action_type": "predict",
                "method": "POST",
                "headers": {
                    "content-type": "application/json"
                },
                "url": remote_ml[remote_ml_key]["endpoint_url"],
                "pre_process_function": remote_ml[remote_ml_key]["pre_process_fun"],
                "request_body": remote_ml[remote_ml_key]["request_body"],
                #"post_process_function": remote_ml[remote_ml_key]["post_process_fun"]
            }
        ]
        }

        r_1 = requests.post(connector_path_url, auth=awsauth, json=payload_1, headers=headers)
        #print(r_1.text)
        remote_ml[remote_ml_key]["connector_id"] = json.loads(r_1.text)["connector_id"]
        
        st.session_state[remote_ml_key+"_CONNNECTOR_ID"] = json.loads(r_1.text)["connector_id"]
        ds.store_in_dynamo(remote_ml_key+"_CONNNECTOR_ID",json.loads(r_1.text)["connector_id"] )
        
        
        
    
    

connector_res = json.loads((requests.post(host+'/_plugins/_ml/connectors/_search',json = {"query": {"match_all": {}}}, auth=awsauth,headers=headers)).text) 

print(connector_res)

if(connector_res["hits"]["total"]["value"] == 0):
    create_ml_connectors()
    

def ingest_data(col,warning):
    
    
    ingest_flag = False
    
     
    opensearch_res = (requests.get(host+'_ingest/pipeline/ml_ingest_pipeline', auth=awsauth,headers=headers)).text
    if('403' in str(opensearch_res)):
        st.error('Please add the IAM role, "'+st.session_state.WebappRoleArn+'" to OpenSearch backend roles [here](%s)' % dash_url ,icon = "🚨")

        return ""
    print("opensearch_res:"+opensearch_res)
    search_types = 'Keyword Search,'
    if(opensearch_res!='{}'):
        #print("------------------"+opensearch_res)
        opensearch_models = {}
        for i in json.loads(opensearch_res)['ml_ingest_pipeline']['processors']:
            key_ = list(i.keys())[0]
            opensearch_models[list(i.keys())[0]] = i[key_]['model_id']
                
            if(key_ == 'sparse_encoding'):
                search_types+='NeuralSparse Search,'
                st.session_state.SAGEMAKER_SPARSE_MODEL_ID = i[key_]['model_id']
                ds.store_in_dynamo('SAGEMAKER_SPARSE_MODEL_ID',st.session_state.SAGEMAKER_SPARSE_MODEL_ID )
        
        
        search_types = search_types[0:-1]
        
        print(opensearch_models)
        response = ds.get_from_dynamo("ml_ingest_pipeline")
        if(response == ""):
            dynamo_res = '{}'
            ds.store_in_dynamo('ml_ingest_pipeline',opensearch_res)
            
            ingest_flag = True
            
            
        else:
            dynamo_res = json.loads(response)
            dynamo_models = {}
            for i in dynamo_res['ml_ingest_pipeline']['processors']:
                key_ = list(i.keys())[0]
                dynamo_models[list(i.keys())[0]] = i[key_]['model_id']
            if(opensearch_models!=dynamo_models):
                ds.update_in_dynamo('ml_ingest_pipeline','store_val',opensearch_res)
                ingest_flag = True
    else:
        #print("------------------"+opensearch_res)
        exists = requests.head(host+'demostore-search-index', auth=awsauth,headers=headers)
        if(str(exists) == '<Response [404]>'):
            ingest_flag = True
    print(ingest_flag)
    
    ds.store_in_dynamo('search_types',search_types)
    st.session_state.search_types = search_types
    
            
    if(ingest_flag == False):
        return ""
    
    with warning:
        st.warning("Please wait while the data is ingested. Do not refresh the page !",icon = "⚠️")
    
    aos_client = OpenSearch(
    hosts = [{'host': OpenSearchDomainEndpoint, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    connection_class = RequestsHttpConnection
        )
    
    
    yaml = YAML()
    items_ = yaml.load(open('/home/ec2-user/SageMaker/images_retail/products.yaml'))

    batch = 0
    count = 0
    body_ = ''
    batch_size = 50
    last_batch = int(len(items_)/batch_size)
    action = json.dumps({ 'index': { '_index': 'demostore-search-index' } })
  


         
    for item in items_:
        count+=1
        fileshort = "/home/ec2-user/SageMaker/images_retail/"+item["category"]+"/"+item["image"]
        payload = {}
        payload['image_url'] = fileshort
        payload['product_description'] = item['description']
        payload['caption'] = item['name']
        payload['category'] = item['category']
        payload['price'] = item['price']
        if('gender_affinity' in item):
            if(item['gender_affinity'] == 'M'):
                payload['gender_affinity'] = 'Male'
            else:
                if(item['gender_affinity'] == 'F'):
                    payload['gender_affinity'] = 'Female'
                else:
                    payload['gender_affinity'] = payload['gender_affinity']
        else:
            payload['gender_affinity'] = ""
        if('style' in item):
            payload['style'] = item['style']
        else:
            payload['style'] = ""
        
        
        
        body_ = body_ + action + "\n" + json.dumps(payload) + "\n"
        
        if(count == batch_size):
            response = aos_client.bulk(
            index = 'demostore-search-index',
            body = body_
            )
            batch += 1
            count = 0
            print("batch "+str(batch) + " ingestion done!")
            
            if(batch != last_batch):
                body_ = ""
            
            
                
            #ingest the remaining rows
    response = aos_client.bulk(
            index = 'demostore-search-index',
            body = body_
            )
    print(response)                
    print("All "+str(last_batch)+" batches ingested into index")
    warning.empty()
    with col:
        st.write(":white_check_mark:")
    
    
    
    
    
    
    
    
    

    
    #Enable Playground
    
    st.session_state.play_disabled = 'False'
    ds.store_in_dynamo('play_disabled','False')
    
    
    
    
                
    #st.switch_page('pages/Semantic_Search.py')





# input_host = st.text_input( "Your OpenSearch domain URL",key="input_host",placeholder = "Opensearch host",value = "https://search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com/")
# input_index = st.text_input( "Your OpenSearch domain index",key="input_index",placeholder = "Opensearch index name", value = "raw-retail-ml-search-index")
input_host = "https://search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com/"
input_index = "raw-retail-ml-search-index"
url = input_host + input_index
# get_fileds = st.button('Get field metadata')
st.write("----",divider = "rainbow")
warning = st.empty()
c1,c2,c3,c4 = st.columns([25,25,25,25])
with c2:
    inner_col1,inner_col2 = st.columns([55,70])
    with inner_col1:
        ingest_data = st.button('(Re)Index data',type = 'primary',on_click = ingest_data, args=(inner_col2,warning))

print("st.session_state.play_disabled")
print(st.session_state.play_disabled )
if(st.session_state.play_disabled == "" or st.session_state.play_disabled == "True"):
    st.session_state.play_disabled  = True
else:
    st.session_state.play_disabled  = False
    
print("st.session_state.play_disabled")
print(st.session_state.play_disabled)

with c3:
    playground = st.button('Launch playground', type = 'primary', disabled = st.session_state.play_disabled)#st.session_state.play_disabled
if(playground):
    st.switch_page('pages/Semantic_Search.py')

            