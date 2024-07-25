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

st.set_page_config(
    
    #page_title="Semantic Search using OpenSearch",
    layout="wide",
    page_icon="/home/ubuntu/images/opensearch_mark_default.png"
)

if "play_disabled" not in st.session_state:
    st.session_state.play_disabled = True
    
if "index_map" not in st.session_state:
    st.session_state.index_map = {}
    
if "OpenSearchDomainEndpoint" not in st.session_state:
    st.session_state.OpenSearchDomainEndpoint = ""
    
if "BEDROCK_MULTIMODAL_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_MULTIMODAL_MODEL_ID = ""


if "SAGEMAKER_SPARSE_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_SPARSE_MODEL_ID = ""    
    
if "BEDROCK_TEXT_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_TEXT_MODEL_ID = "" 
    
    

    
if(os.path.isdir("/home/ec2-user/images_retail") == False):

    metadata_file = urllib.request.urlretrieve('https://aws-blogs-artifacts-public.s3.amazonaws.com/BDB-3144/products-data.yml', '/home/ec2-user/images_retail/products.yaml')
    img_filename,headers= urllib.request.urlretrieve('https://aws-blogs-artifacts-public.s3.amazonaws.com/BDB-3144/images.tar.gz', '/home/ec2-user/images_retail/images.tar.gz')              
    print(img_filename)
    file = tarfile.open('/home/ec2-user/images_retail/images.tar.gz')
    file.extractall('/home/ec2-user/images_retail')
    file.close()
    #remove images.tar.gz
    os.remove('/home/ec2-user/images_retail/images.tar.gz')
    
cfn = boto3.client('cloudformation',region_name='us-east-1')

response = cfn.list_stacks(StackStatusFilter=['CREATE_COMPLETE','UPDATE_COMPLETE'])

for cfns in response['StackSummaries']:
    if('TemplateDescription' in cfns.keys()):
        if('hybrid search' in cfns['TemplateDescription']):
            stackname = cfns['StackName']


response = cfn.describe_stack_resources(
    StackName=stackname
)


cfn_outputs = cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']

for output in cfn_outputs:
    if('OpenSearchDomainEndpoint' in output['OutputKey']):
        OpenSearchDomainEndpoint = output['OutputValue']
        
    SparseEmbeddingEndpointName = "neural-sparse"
    
    if('SparseEmbeddingEndpointName' in output['OutputKey']):
        SparseEmbeddingEndpointName = output['OutputValue']
        
    if('s3' in output['OutputKey'].lower()):
        s3_bucket = output['OutputValue']
        

region = 'us-east-1'#boto3.Session().region_name  
print(region)
        

account_id = boto3.client('sts').get_caller_identity().get('Account')



print("stackname: "+stackname)
print("account_id: "+account_id)  
#print("region: "+region)
print("SparseEmbeddingEndpointName: "+SparseEmbeddingEndpointName)
print("OpenSearchDomainEndpoint: "+OpenSearchDomainEndpoint)
print("S3 Bucket: "+s3_bucket)
st.session_state.OpenSearchDomainEndpoint = OpenSearchDomainEndpoint
def create_ml_components():
    
    
    
    # 2. Create the OpenSearch-Sagemaker ML connector
    
    host = 'https://'+OpenSearchDomainEndpoint+'/'
    service = 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)


    remote_ml = {
                "sagemaker_sparse":
                 {
                     "endpoint_url":"https://runtime.sagemaker."+region+".amazonaws.com/endpoints/"+SparseEmbeddingEndpointName+"/invocations",
                     "pre_process_fun": '\n    StringBuilder builder = new StringBuilder();\n    builder.append("\\"");\n    builder.append(params.text_docs[0]);\n    builder.append("\\"");\n    def parameters = "{" +"\\"inputs\\":" + builder + "}";\n    return "{" +"\\"parameters\\":" + parameters + "}";\n    ', 
                   #"post_process_fun": '\n    def name = "sentence_embedding";\n    def dataType = "FLOAT32";\n    if (params.result == null || params.result.length == 0) {\n        return null;\n    }\n    def shape = [params.result[0].length];\n    def json = "{" +\n               "\\"name\\":\\"" + name + "\\"," +\n               "\\"data_type\\":\\"" + dataType + "\\"," +\n               "\\"shape\\":" + shape + "," +\n               "\\"data\\":" + params.result[0] +\n               "}";\n    return json;\n    ',
                    "request_body": """["${parameters.inputs}"]"""
             
                 },
                
                 "bedrock_text":
                {
                     "endpoint_url":"https://bedrock-runtime."+region+".amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
                    "pre_process_fun": "\n    StringBuilder builder = new StringBuilder();\n    builder.append(\"\\\"\");\n    String first = params.text_docs[0];\n    builder.append(first);\n    builder.append(\"\\\"\");\n    def parameters = \"{\" +\"\\\"inputText\\\":\" + builder + \"}\";\n    return  \"{\" +\"\\\"parameters\\\":\" + parameters + \"}\";",
      
                    "post_process_fun":'\n    def name = "sentence_embedding";\n    def dataType = "FLOAT32";\n    if (params.embedding == null || params.embedding.length == 0) {\n        return null;\n    }\n    def shape = [params.embedding.length];\n    def json = "{" +\n               "\\"name\\":\\"" + name + "\\"," +\n               "\\"data_type\\":\\"" + dataType + "\\"," +\n               "\\"shape\\":" + shape + "," +\n               "\\"data\\":" + params.embedding +\n               "}";\n    return json;\n    ',
                    "request_body": "{ \"inputText\": \"${parameters.inputText}\"}"
                 },
                
                 "bedrock_multimodal":
                {
                     "endpoint_url": "https://bedrock-runtime."+region+".amazonaws.com/model/amazon.titan-embed-image-v1/invoke",
                     "request_body": "{ \"inputText\": \"${parameters.inputText:-null}\", \"inputImage\": \"${parameters.inputImage:-null}\" }",
                      "pre_process_fun": "\n    StringBuilder parametersBuilder = new StringBuilder(\"{\");\n    if (params.text_docs.length > 0 && params.text_docs[0] != null) {\n      parametersBuilder.append(\"\\\"inputText\\\":\");\n      parametersBuilder.append(\"\\\"\");\n      parametersBuilder.append(params.text_docs[0]);\n      parametersBuilder.append(\"\\\"\");\n      \n      if (params.text_docs.length > 1 && params.text_docs[1] != null) {\n        parametersBuilder.append(\",\");\n      }\n    }\n    \n    \n    if (params.text_docs.length > 1 && params.text_docs[1] != null) {\n      parametersBuilder.append(\"\\\"inputImage\\\":\");\n      parametersBuilder.append(\"\\\"\");\n      parametersBuilder.append(params.text_docs[1]);\n      parametersBuilder.append(\"\\\"\");\n    }\n    parametersBuilder.append(\"}\");\n    \n    return  \"{\" +\"\\\"parameters\\\":\" + parametersBuilder + \"}\";",
                     "post_process_fun":'\n    def name = "sentence_embedding";\n    def dataType = "FLOAT32";\n    if (params.embedding == null || params.embedding.length == 0) {\n        return null;\n    }\n    def shape = [params.embedding.length];\n    def json = "{" +\n               "\\"name\\":\\"" + name + "\\"," +\n               "\\"data_type\\":\\"" + dataType + "\\"," +\n               "\\"shape\\":" + shape + "," +\n               "\\"data\\":" + params.embedding +\n               "}";\n    return json;\n    '
                    }
            }

    connector_path_url = host+'_plugins/_ml/connectors/_create'
    register_model_path_url = host+'_plugins/_ml/models/_register'


    headers = {"Content-Type": "application/json"}

    for remote_ml_key in remote_ml.keys():
        
        #create connector
        payload_1 = {
        "name": remote_ml_key+": embedding",
        "description": "Test connector for"+remote_ml_key+" remote embedding model",
        "version": 1,
        "protocol": "aws_sigv4",
        "credential": {
            "roleArn": "arn:aws:iam::"+account_id+":role/opensearch-sagemaker-role"
        },
        "parameters": {
            "region": region,
            "service_name": remote_ml_key.split("_")[0]
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
        if(remote_ml_key != 'sagemaker_sparse'):
            payload_1["actions"][0]["post_process_function"] = remote_ml[remote_ml_key]["post_process_fun"]
            
        

        r_1 = requests.post(connector_path_url, auth=awsauth, json=payload_1, headers=headers)
        print(r_1.text)
        remote_ml[remote_ml_key]["connector_id"] = json.loads(r_1.text)["connector_id"]
        
        time.sleep(2)
        
        #register model
        
        payload_2 = { 
                    "name": remote_ml_key,
                    "function_name":"remote",
                    "description": remote_ml_key+" embeddings model",
                    "connector_id": remote_ml[remote_ml_key]["connector_id"]
                    
                    }

        r_2 = requests.post(register_model_path_url, auth=awsauth, json=payload_2, headers=headers)
        remote_ml[remote_ml_key]["model_id"] = json.loads(r_2.text)["model_id"]
        
        time.sleep(2)
        
        #deploy model
        
        deploy_model_path_url = host+'_plugins/_ml/models/'+remote_ml[remote_ml_key]["model_id"]+'/_deploy'

        r_3 = requests.post(deploy_model_path_url, auth=awsauth, headers=headers)
        deploy_status = json.loads(r_3.text)["status"]
        print("Deployment status of the "+remote_ml_key+" model, "+remote_ml[remote_ml_key]["model_id"]+" : "+deploy_status)
        
        time.sleep(2)
        #test model
        payload_4 = {
                    "parameters": {
                        "inputText": "hello",
                        "inputImage":'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'
                        }
                            }
        
        if(remote_ml_key == 'sagemaker_sparse'):
            payload_4 = {
        "parameters": {
            "inputs": "hello"
            }
                }

        path_4 = host+'_plugins/_ml/models/'+remote_ml[remote_ml_key]["model_id"]+'/_predict'
        r_4 = requests.post(path_4, auth=awsauth, json=payload_4, headers=headers)
        print(r_4.text)
        if(remote_ml_key != 'sagemaker_sparse'):
            embed = json.loads(r_4.text)['inference_results'][0]['output'][0]['data'][0:4]
            shape = json.loads(r_4.text)['inference_results'][0]['output'][0]['shape']
            remote_ml[remote_ml_key]['dimensions'] = shape[0]
            print(remote_ml_key+ " : "+str(embed))
            print(shape)
            print("\n")
    
    
    # 3. Model IDs assignment
    
    REGION = "us-east-1" 
    SAGEMAKER_SPARSE_MODEL_ID = remote_ml["sagemaker_sparse"]["model_id"] 
    st.session_state.SAGEMAKER_SPARSE_MODEL_ID = SAGEMAKER_SPARSE_MODEL_ID
    BEDROCK_TEXT_MODEL_ID = remote_ml["bedrock_text"]["model_id"]#'iUvGQYwBuQkLO8mDfE0l'#
    st.session_state.BEDROCK_TEXT_MODEL_ID = BEDROCK_TEXT_MODEL_ID
    BEDROCK_MULTIMODAL_MODEL_ID = remote_ml["bedrock_multimodal"]["model_id"]#'o6GykYwB08ySW4fgXdf3'#
    st.session_state.BEDROCK_MULTIMODAL_MODEL_ID = BEDROCK_MULTIMODAL_MODEL_ID
    
    
    
    # 4. Create ingest pipelines
    
    pipeline = "ml-ingest-pipeline"
    vector_field = st.session_state.input_vectors
    
    pipeline_payload = {

            "description": "ML ingest pipeline",
            "processors": [
            {
                "text_embedding": {
                "model_id": BEDROCK_TEXT_MODEL_ID,
                "field_map": {
                    vector_field: vector_field+"_embedding_bedrock-text"
                }
                }
            },
            {
                "text_image_embedding": {
                "model_id": BEDROCK_MULTIMODAL_MODEL_ID,
                "embedding": vector_field+"_embedding_bedrock-multimodal",
                "field_map": {
                    "text": vector_field,
                    "image": "image_binary"
                }
                }
            }
            ,
            {
                "sparse_encoding": {
                "model_id": "srrJ-owBQhe1aB-khx2n",
                "field_map": {
                    vector_field: vector_field+"_embedding_sparse"
                }
                }
            }
            ]
        }
    
    pipeline_path = host+'_ingest/pipeline/'+pipeline
    res = requests.put(pipeline_path, auth=awsauth, json=pipeline_payload, headers=headers)   
    print(res.text)
    # 5. Create index
    
    ml_index = "retail-ml-search-index"
    index_path = host+ml_index
    index_payload = {
         "settings": {
    "index.knn": True,
    #"default_pipeline": pipeline
  },
         "mappings":{"properties": st.session_state.index_map}
    }
    res = requests.put(index_path, auth=awsauth, json=index_payload, headers=headers)   
    print(res.text)
    
    
    
    
    # 6. Index data
    
    
    reindex_payload = {
    "source":{
        "remote":{
            "host":st.session_state.input_host[:-1]+":443",
            "region": "us-east-1",
            "username":"test",
            "password":"@ML-Search123"
        },
        "index": "retail-ml-search-index"
    },
    "dest":{
        "index":"retail-ml-search-index"
    }
    }
    reindex_path = host+"_reindex"
    res = requests.post(reindex_path, auth=awsauth, json=reindex_payload, headers=headers)   
    print(res.text)
    
    
    
    
    

    
    
    
    # 7. Enable Playground
    
    st.session_state.play_disabled = False
    
    
    
    
                
    #st.switch_page('pages/Semantic_Search.py')



headers = {"Content-Type": "application/json"}

input_host = st.text_input( "OpenSearch domain URL",key="input_host",placeholder = "Opensearch host",value = "https://search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com/")
input_index = st.text_input( "OpenSearch domain index",key="input_index",placeholder = "Opensearch index name", value = "raw-retail-ml-search-index")
url = input_host + input_index
get_fileds = st.button('Get field metadata')
playground = st.button('Launch Playground', disabled = st.session_state.play_disabled)#st.session_state.play_disabled
if(playground):
    st.switch_page('pages/Semantic_Search.py')
if(get_fileds):
    #DOMAIN_ENDPOINT =   "search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com" #"search-opensearchservi-rimlzstyyeih-3zru5p2nxizobaym45e5inuayq.us-west-2.es.amazonaws.com" 
    REGION = "us-east-1" #'us-west-2'#
    credentials = boto3.Session().get_credentials()
    #awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, 'es', session_token=credentials.token)
    awsauth = HTTPBasicAuth("prasadnu","@Palamalai1")
    r = requests.get(url, auth=awsauth,  headers=headers)
    
    mappings= json.loads(r.text)[input_index]["mappings"]["properties"]
    
    fields = []
    props = {}
    for i in mappings.keys():
        if(mappings[i]["type"] != 'knn_vector' and mappings[i]["type"] != "rank_features"):
            fields.append({i:mappings[i]["type"]})
        if('fields' in mappings[i]):
            del mappings[i]['fields']
    st.session_state.index_map = mappings
    print(st.session_state.index_map)
    col1,col2 = st.columns([50,50])
    with col1:
        st.write(fields)
    with col2:
        input_vectors = st.text_input( "comma separated field names that needs to be vectorised",key="input_vectors",placeholder = "field1,field2")
        submit = st.button("Submit",on_click = create_ml_components)
        
        
    