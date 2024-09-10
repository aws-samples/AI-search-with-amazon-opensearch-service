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
    
if "REGION" not in st.session_state:
    st.session_state.REGION = ""
    
if "BEDROCK_MULTIMODAL_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_MULTIMODAL_MODEL_ID = ds.get_from_dynamo("BEDROCK_MULTIMODAL_MODEL_ID")
    
if "max_selections" not in st.session_state:
    st.session_state.max_selections = ds.get_from_dynamo("max_selections")
    
if "search_types" not in st.session_state:
    st.session_state.search_types =ds.get_from_dynamo("search_types")
    if(st.session_state.search_types == ""):
        st.session_state.search_types = "Keyword Search"

if "SAGEMAKER_SPARSE_MODEL_ID" not in st.session_state:
    st.session_state.SAGEMAKER_SPARSE_MODEL_ID = ds.get_from_dynamo("SAGEMAKER_SPARSE_MODEL_ID")   
    
if "BEDROCK_TEXT_MODEL_ID" not in st.session_state:
    st.session_state.BEDROCK_TEXT_MODEL_ID = ds.get_from_dynamo("BEDROCK_TEXT_MODEL_ID")  
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
        
    
        
        
        


        

account_id = boto3.client('sts').get_caller_identity().get('Account')





print("stackname: "+stackname)
print("account_id: "+account_id)  
#print("region: "+region)
print("OpenSearchDomainEndpoint: "+OpenSearchDomainEndpoint)

st.session_state.OpenSearchDomainEndpoint = OpenSearchDomainEndpoint
ds.store_in_dynamo('OpenSearchDomainEndpoint',st.session_state.OpenSearchDomainEndpoint )
ds.store_in_dynamo('REGION',st.session_state.REGION )


host = 'https://'+OpenSearchDomainEndpoint+'/'
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, st.session_state.REGION, service, session_token=credentials.token)
headers = {"Content-Type": "application/json"}

exists_ = requests.head(host+'demostore-search-index', auth=awsauth,headers=headers)
print(exists_)
if('403' in str(exists_)):
        st.error("Please add the backend role in OpenSearch dashboards",icon = "🚨")
   
if('404' in str(exists_) or '403' in str(exists_)):
    
    st.session_state.play_disabled = 'True'
else:
    st.session_state.play_disabled = 'False'
ds.store_in_dynamo('play_disabled',st.session_state.play_disabled)
    

opensearch_search_pipeline = (requests.get(host+'_search/pipeline/ml_search_pipeline', auth=awsauth,headers=headers)).text
print("opensearch_search_pipeline")
print(opensearch_search_pipeline)
if(opensearch_search_pipeline!='{}'):
    st.session_state.max_selections = "None"
else:
    st.session_state.max_selections = "1"
ds.store_in_dynamo('max_selections',st.session_state.max_selections )

def ingest_data(col,default):
    
    
    ingest_flag = False
    
     
    opensearch_res = (requests.get(host+'_ingest/pipeline/ml_ingest_pipeline', auth=awsauth,headers=headers)).text
    if('403' in str(opensearch_res)):
        st.error("Please add the backend role in OpenSearch dashboards",icon = "🚨")
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
                    payload['gender_affinity'] = item['gender_affinity']
        if('style' in item):
            payload['style'] = item['style']
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
c1,c2,c3,c4 = st.columns([25,25,25,25])
with c2:
    inner_col1,inner_col2 = st.columns([55,70])
    with inner_col1:
        ingest_data = st.button('(Re)Index data',type = 'primary',on_click = ingest_data, args=(inner_col2,"default"))

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
        
        
    