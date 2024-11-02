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
    
if "re_ranker" not in st.session_state:
    st.session_state.re_ranker = ds.get_from_dynamo("re_ranker")
    
if "search_types" not in st.session_state:
    st.session_state.search_types =ds.get_from_dynamo("search_types")
    if(st.session_state.search_types == ""):
        st.session_state.search_types = "Keyword Search"

if "SAGEMAKER_SPARSE_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_SPARSE_MODEL_ID = ds.get_from_dynamo("SAGEMAKER_SPARSE_MODEL_ID")   
    
if "SAGEMAKER_SPARSE_CONNECTOR_ID" not in st.session_state:
    st.session_state.SAGEMAKER_SPARSE_CONNECTOR_ID = ds.get_from_dynamo("SAGEMAKER_SPARSE_CONNECTOR_ID")   
    
    
if "SAGEMAKER_CrossEncoder_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID = ds.get_from_dynamo("SAGEMAKER_CrossEncoder_MODEL_ID")   
    
if "SAGEMAKER_CrossEncoder_CONNECTOR_ID" not in st.session_state:
    st.session_state.SAGEMAKER_CrossEncoder_CONNECTOR_ID = ds.get_from_dynamo("SAGEMAKER_CrossEncoder_CONNECTOR_ID")  
    
if "BEDROCK_TEXT_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_TEXT_MODEL_ID = ds.get_from_dynamo("BEDROCK_TEXT_MODEL_ID")  
    
if "BEDROCK_TEXT_CONNECTOR_ID" not in st.session_state:
    st.session_state.BEDROCK_TEXT_CONNECTOR_ID = ds.get_from_dynamo("BEDROCK_TEXT_CONNECTOR_ID")  
#bytes_for_rekog = ""
    
    

    

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
    
###### check for search pipelines #######
opensearch_search_pipeline = (requests.get(host+'_search/pipeline/hybrid_search_pipeline', auth=awsauth,headers=headers)).text
print("opensearch_search_pipeline")
print(opensearch_search_pipeline)
if(opensearch_search_pipeline!='{}'):
    st.session_state.max_selections = "None"
else:
    st.session_state.max_selections = "1"
        
ds.store_in_dynamo('max_selections',st.session_state.max_selections )
        
opensearch_rerank_pipeline = (requests.get(host+'_search/pipeline/rerank_pipeline', auth=awsauth,headers=headers)).text
print("opensearch_rerank_pipeline")
print(opensearch_rerank_pipeline)
if(opensearch_rerank_pipeline!='{}'):
    st.session_state.re_ranker = "true"
    total_pipeline = json.loads(opensearch_rerank_pipeline)
    if('response_processors' in total_pipeline['rerank_pipeline'].keys()):
        st.session_state.re_ranker = "true"
        st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID = total_pipeline['rerank_pipeline']['response_processors'][0]['rerank']['ml_opensearch']['model_id']
        ds.store_in_dynamo('SAGEMAKER_CrossEncoder_MODEL_ID',st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID )
        
else:
    st.session_state.re_ranker = "false"
ds.store_in_dynamo('re_ranker',st.session_state.re_ranker )
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
             
                 },
        "SAGEMAKER_CrossEncoder":
                 {
                     
                      "endpoint_url": "https://runtime.sagemaker."+st.session_state.REGION+".amazonaws.com/endpoints/cross-encoder-model/invocations",
            "request_body": "{ \"inputs\": ${parameters.inputs} }",
      "pre_process_fun": "\n    String escape(def input) { \n       if (input.contains(\"\\\\\")) {\n        input = input.replace(\"\\\\\", \"\\\\\\\\\");\n      }\n      if (input.contains(\"\\\"\")) {\n        input = input.replace(\"\\\"\", \"\\\\\\\"\");\n      }\n      if (input.contains('\r')) {\n        input = input = input.replace('\r', '\\\\r');\n      }\n      if (input.contains(\"\\\\t\")) {\n        input = input.replace(\"\\\\t\", \"\\\\\\\\\\\\t\");\n      }\n      if (input.contains('\n')) {\n        input = input.replace('\n', '\\\\n');\n      }\n      if (input.contains('\b')) {\n        input = input.replace('\b', '\\\\b');\n      }\n      if (input.contains('\f')) {\n        input = input.replace('\f', '\\\\f');\n      }\n      return input;\n    }\n\n   String query = params.query_text;\n   StringBuilder builder = new StringBuilder('[');\n    \n    for (int i=0; i<params.text_docs.length; i ++) {\n      builder.append('{\"text\":\"');\n      builder.append(escape(query));\n      builder.append('\", \"text_pair\":\"');\n      builder.append(escape(params.text_docs[i]));\n      builder.append('\"}');\n      if (i<params.text_docs.length - 1) {\n        builder.append(',');\n      }\n    }\n    builder.append(']');\n    \n    def parameters = '{ \"inputs\": ' + builder + ' }';\n    return  '{\"parameters\": ' + parameters + '}';\n     ",
      "post_process_fun": "\n      \n      def dataType = \"FLOAT32\";\n      \n      \n      if (params.result == null)\n      {\n          return 'no result generated';\n          //return params.response;\n      }\n      def outputs = params.result;\n      \n      \n      def resultBuilder = new StringBuilder('[ ');\n      for (int i=0; i<outputs.length; i++) {\n        resultBuilder.append(' {\"name\": \"similarity\", \"data_type\": \"FLOAT32\", \"shape\": [1],');\n        //resultBuilder.append('{\"name\": \"similarity\"}');\n        \n        resultBuilder.append('\"data\": [');\n        resultBuilder.append(outputs[i].score);\n        resultBuilder.append(']}');\n        if (i<outputs.length - 1) {\n          resultBuilder.append(',');\n        }\n      }\n      resultBuilder.append(']');\n      \n      return resultBuilder.toString();\n    "
  
             
                 },
                
                 "BEDROCK_TEXT":
                {
                     "endpoint_url":"https://bedrock-runtime."+st.session_state.REGION+".amazonaws.com/model/amazon.titan-embed-text-v1/invoke",
                    "pre_process_fun": "\n    StringBuilder builder = new StringBuilder();\n    builder.append(\"\\\"\");\n    String first = params.text_docs[0];\n    builder.append(first);\n    builder.append(\"\\\"\");\n    def parameters = \"{\" +\"\\\"inputText\\\":\" + builder + \"}\";\n    return  \"{\" +\"\\\"parameters\\\":\" + parameters + \"}\";",
      
                    "post_process_fun":'\n    def name = "sentence_embedding";\n    def dataType = "FLOAT32";\n    if (params.embedding == null || params.embedding.length == 0) {\n        return null;\n    }\n    def shape = [params.embedding.length];\n    def json = "{" +\n               "\\"name\\":\\"" + name + "\\"," +\n               "\\"data_type\\":\\"" + dataType + "\\"," +\n               "\\"shape\\":" + shape + "," +\n               "\\"data\\":" + params.embedding +\n               "}";\n    return json;\n    ',
                    "request_body": "{ \"inputText\": \"${parameters.inputText}\"}"
                 },
                
                 "BEDROCK_MULTIMODAL":
                {
                     "endpoint_url": "https://bedrock-runtime."+st.session_state.REGION+".amazonaws.com/model/amazon.titan-embed-image-v1/invoke",
                     "request_body": "{ \"inputText\": \"${parameters.inputText:-null}\", \"inputImage\": \"${parameters.inputImage:-null}\" }",
                      "pre_process_fun": "\n    StringBuilder parametersBuilder = new StringBuilder(\"{\");\n    if (params.text_docs.length > 0 && params.text_docs[0] != null) {\n      parametersBuilder.append(\"\\\"inputText\\\":\");\n      parametersBuilder.append(\"\\\"\");\n      parametersBuilder.append(params.text_docs[0]);\n      parametersBuilder.append(\"\\\"\");\n      \n      if (params.text_docs.length > 1 && params.text_docs[1] != null) {\n        parametersBuilder.append(\",\");\n      }\n    }\n    \n    \n    if (params.text_docs.length > 1 && params.text_docs[1] != null) {\n      parametersBuilder.append(\"\\\"inputImage\\\":\");\n      parametersBuilder.append(\"\\\"\");\n      parametersBuilder.append(params.text_docs[1]);\n      parametersBuilder.append(\"\\\"\");\n    }\n    parametersBuilder.append(\"}\");\n    \n    return  \"{\" +\"\\\"parameters\\\":\" + parametersBuilder + \"}\";",
                     "post_process_fun":'\n    def name = "sentence_embedding";\n    def dataType = "FLOAT32";\n    if (params.embedding == null || params.embedding.length == 0) {\n        return null;\n    }\n    def shape = [params.embedding.length];\n    def json = "{" +\n               "\\"name\\":\\"" + name + "\\"," +\n               "\\"data_type\\":\\"" + dataType + "\\"," +\n               "\\"shape\\":" + shape + "," +\n               "\\"data\\":" + params.embedding +\n               "}";\n    return json;\n    '
                    }
            }

    connector_path_url = host+'_plugins/_ml/connectors/_create'
    

    for remote_ml_key in remote_ml.keys():
        
        #create connector
        payload_1 = {
        "name": remote_ml_key+": EMBEDDING",
        "description": "Test connector for"+remote_ml_key+" remote embedding model",
        "version": 1,
        "protocol": "aws_sigv4",
        "credential": {
            "roleArn": "arn:aws:iam::"+account_id+":role/opensearch-sagemaker-role"
        },
        "parameters": {
            "region": st.session_state.REGION,
            "service_name": (remote_ml_key.split("_")[0]).lower()
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
        if(remote_ml_key != 'SAGEMAKER_SPARSE'):
            payload_1["actions"][0]["post_process_function"] = remote_ml[remote_ml_key]["post_process_fun"]
            
        

        r_1 = requests.post(connector_path_url, auth=awsauth, json=payload_1, headers=headers)
        print(r_1.text)
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
        opensearch_models = {}
        for i in json.loads(opensearch_res)['ml_ingest_pipeline']['processors']:
            key_ = list(i.keys())[0]
            opensearch_models[list(i.keys())[0]] = i[key_]['model_id']
            if(key_ == 'text_embedding'):
                search_types+='Vector Search,'
                st.session_state.BEDROCK_TEXT_MODEL_ID = i[key_]['model_id']
                ds.store_in_dynamo('BEDROCK_TEXT_MODEL_ID',st.session_state.BEDROCK_TEXT_MODEL_ID )
                
            if(key_ == 'text_image_embedding'):
                search_types+='Multimodal Search,'
                st.session_state.BEDROCK_MULTIMODAL_MODEL_ID = i[key_]['model_id']
                ds.store_in_dynamo('BEDROCK_MULTIMODAL_MODEL_ID',st.session_state.BEDROCK_MULTIMODAL_MODEL_ID )
                
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
        exists = requests.head(host+'demostore-search-index', auth=awsauth,headers=headers)
        if(str(exists) == '<Response [404]>'):
            ingest_flag = True
    print(ingest_flag)
    
    ds.store_in_dynamo('search_types',search_types)
    st.session_state.search_types = search_types
    
            
    if(ingest_flag == False):
        return ""
    
    
    
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
    
    from PIL import Image
    import base64

    def resize_image(photo, width, height):
        Image.MAX_IMAGE_PIXELS = 100000000
        
        with Image.open(photo) as image:
            image.verify()
        with Image.open(photo) as image:    
            
            if image.format in ["JPEG", "PNG"]:
                file_type = image.format.lower()
                path = image.filename.rsplit(".", 1)[0]

                image.thumbnail((width, height))
                image.save(f"{path}-resized.{file_type}")
        return file_type, path
    with warning:
        st.warning("Please wait while the data is ingested. Do not refresh the page !",icon = "⚠️")
         
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
        #resize the image and generate image binary
        
        file_type, path = resize_image(fileshort, 2048, 2048)

        with open(fileshort.split(".")[0]+"-resized."+file_type, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode("utf8")
    
        os.remove(fileshort.split(".")[0]+"-resized."+file_type)
        payload['product_image'] = input_image
    
        
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
# if(get_fileds):
#     #DOMAIN_ENDPOINT =   "search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com" #"search-opensearchservi-rimlzstyyeih-3zru5p2nxizobaym45e5inuayq.us-west-2.es.amazonaws.com" 
#     REGION = st.session_state.REGION #'us-west-2'#
#     credentials = boto3.Session().get_credentials()
#     #awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, 'es', session_token=credentials.token)
#     awsauth = HTTPBasicAuth(st.session_state.OpenSearchUsername,st.session_state.OpenSearchPassword)
#     r = requests.get(url, auth=awsauth,  headers=headers)
#     mappings= json.loads(r.text)[input_index]["mappings"]["properties"]
    
#     fields = []
#     props = {}
#     for i in mappings.keys():
#         if(mappings[i]["type"] != 'knn_vector' and mappings[i]["type"] != "rank_features"):
#             fields.append({i:mappings[i]["type"]})
#         if('fields' in mappings[i]):
#             del mappings[i]['fields']
#     st.session_state.index_map = mappings
#     print(st.session_state.index_map)
#     col1,col2 = st.columns([50,50])
#     with col1:
#         st.write(fields)
#     with col2:
#         input_vectors = st.text_input( "Field name(s) (comma separated) that needs to be vectorised",key="input_vectors",placeholder = "field1,field2")
#         submit = st.button("Submit",on_click = create_ml_components)
        
        
    