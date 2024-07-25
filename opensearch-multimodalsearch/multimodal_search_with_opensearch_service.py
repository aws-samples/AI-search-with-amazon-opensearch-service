#!/usr/bin/env python
# coding: utf-8

# # Build Multimodal Search with Amazon OpenSearch service

# In[ ]:


#Install python packages
get_ipython().run_line_magic('pip', 'install requests_aws4auth boto3 pillow opensearch_py ipywidgets')


# ## 1. Setup OpenSearch client
# 
# NOTE: Values enclosed within <> are the parameters that you should configure
# 
# You should also setup opensearch authentication in the code 

# In[ ]:


import requests
from requests_aws4auth import AWS4Auth
import boto3
import json
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests.auth import HTTPBasicAuth


headers = {"Content-Type": "application/json"}
host = <opensearch domain endpoint without 'https://'> #example: "opensearch-domain-endpoint.us-east-1.es.amazonaws.com"
service = 'es'
region = <aws-region> #example: "us-east-1"


# 3.Use OpenSearch master credentials that you created while creating the OpenSearch domain
awsauth = HTTPBasicAuth(<master username>,<master password>)


#Initialise OpenSearch-py client
aos_client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    connection_class = RequestsHttpConnection
)


# ## 2. Download the dataset (.gz) and extract the .gz file

# In[ ]:


import os
import urllib.request
import tarfile

os.makedirs('tmp/images', exist_ok = True)
metadata_file = urllib.request.urlretrieve('https://aws-blogs-artifacts-public.s3.amazonaws.com/BDB-3144/products-data.yml', 'tmp/images/products.yaml')
img_filename,headers= urllib.request.urlretrieve('https://aws-blogs-artifacts-public.s3.amazonaws.com/BDB-3144/images.tar.gz', 'tmp/images/images.tar.gz')              
print(img_filename)
file = tarfile.open('tmp/images/images.tar.gz')
file.extractall('tmp/images/')
file.close()
#remove images.tar.gz
os.remove('tmp/images/images.tar.gz')


# # Do you have the model_id already ?

# In[ ]:


import ipywidgets as widgets
from ipywidgets import Dropdown

model_id_selection = [
    "I have the model_id already from the cloudformation",
    "I don't have the model_id already fromthe cloudformation",
]

model_id_dropdown = widgets.Dropdown(
    options=model_id_selection,
    value="I have the model_id already from the cloudformation",
    description="Is model_id already created ?",
    style={"description_width": "initial"},
    layout={"width": "max-content"},
)

display(model_id_dropdown)


# In[ ]:


if(model_id_dropdown.value == "I have the model_id already from the cloudformation"):
    model_id = (input("Enter the model_id and press ENTER : "))
    print("model_id: "+model_id)
else:
    model_id = ""
    print("model_id: '' \nCreate the model id by running next step")


# # Run the below step 3 only if you DO NOT have the model_id 
# 
# # If you have the model_id already created from the cloudformation template, skip step 3 and proceed from step 4

# ## 3. Create the OpenSearch Bedrock ML connector
# 
# you need to change **"iam-role-arn"** below with the ARN of the IAM role that has permissions to talk to OpenSearch and mapped as back-end role in OpenSearch dashboards

# In[ ]:


# Register repository
if(model_id == ''):
    path = '_plugins/_ml/connectors/_create'
    url = 'https://'+host + '/' + path

    payload = {
       "name": "sagemaker: embedding",
       "description": "Test connector for Sagemaker embedding model",
       "version": 1,
       "protocol": "aws_sigv4",
       "credential": {
          "roleArn": "<iam-role-arn>"
       },
       "parameters": {
          "region": region,
          "service_name": "bedrock"
       },
       "actions": [
          {
             "action_type": "predict",
             "method": "POST",
           "headers": {
            "content-type": "application/json",
            "x-amz-content-sha256": "required"
          },

        "url": "https://bedrock-runtime."+region+".amazonaws.com/model/amazon.titan-embed-image-v1/invoke",
         "request_body": "{ \"inputText\": \"${parameters.inputText:-null}\", \"inputImage\": \"${parameters.inputImage:-null}\" }",
          "pre_process_function": "\n    StringBuilder parametersBuilder = new StringBuilder(\"{\");\n    if (params.text_docs.length > 0 && params.text_docs[0] != null) {\n      parametersBuilder.append(\"\\\"inputText\\\":\");\n      parametersBuilder.append(\"\\\"\");\n      parametersBuilder.append(params.text_docs[0]);\n      parametersBuilder.append(\"\\\"\");\n      \n      if (params.text_docs.length > 1 && params.text_docs[1] != null) {\n        parametersBuilder.append(\",\");\n      }\n    }\n    \n    \n    if (params.text_docs.length > 1 && params.text_docs[1] != null) {\n      parametersBuilder.append(\"\\\"inputImage\\\":\");\n      parametersBuilder.append(\"\\\"\");\n      parametersBuilder.append(params.text_docs[1]);\n      parametersBuilder.append(\"\\\"\");\n    }\n    parametersBuilder.append(\"}\");\n    \n    return  \"{\" +\"\\\"parameters\\\":\" + parametersBuilder + \"}\";",
          "post_process_function": "\n      def name = \"sentence_embedding\";\n      def dataType = \"FLOAT32\";\n      if (params.embedding == null || params.embedding.length == 0) {\n          return null;\n      }\n      def shape = [params.embedding.length];\n      def json = \"{\" +\n                 \"\\\"name\\\":\\\"\" + name + \"\\\",\" +\n                 \"\\\"data_type\\\":\\\"\" + dataType + \"\\\",\" +\n                 \"\\\"shape\\\":\" + shape + \",\" +\n                 \"\\\"data\\\":\" + params.embedding +\n                 \"}\";\n      return json;\n    "
          }
       ]
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, auth=awsauth, json=payload, headers=headers)
    print(r.status_code)
    print(r.text)
    connector_id = json.loads(r.text)["connector_id"]

    # Register the model
    path = '_plugins/_ml/models/_register'
    url = 'https://'+host + '/' + path
    payload = { "name": "Bedrock Multimodal embeddings model",
    "function_name": "remote",
    "description": "Bedrock Multimodal embeddings model",
    "connector_id": connector_id}
    r = requests.post(url, auth=awsauth, json=payload, headers=headers)
    model_id = json.loads(r.text)["model_id"]
    print("Model registered under model_id: "+model_id)

    # Deploy the model
    path = '_plugins/_ml/models/'+model_id+'/_deploy'
    url = 'https://'+host + '/' + path
    r = requests.post(url, auth=awsauth, headers=headers)
    deploy_status = json.loads(r.text)["status"]
    print("Deployment status of the model, "+model_id+" : "+deploy_status)


# ## 4. Test the OpenSearch - Bedrock integration with a test input

# In[ ]:


import base64

path = '_plugins/_ml/models/'+model_id+'/_predict'
url = 'https://'+host + '/' + path
img = "tmp/images/footwear/2d2d8ec8-4806-42a7-b8ba-ceb15c1c7e84.jpg"
with open(img, "rb") as image_file:
    input_image_binary = base64.b64encode(image_file.read()).decode("utf8")
    
payload = {
"parameters": {
"inputText": "Sleek, stylish black sneakers made for urban exploration. With fashionable looks and comfortable design, these sneakers keep your feet looking great while you walk the city streets in style",
"inputImage":input_image_binary
}
}
r = requests.post(url, auth=awsauth, json=payload, headers=headers)
embed = json.loads(r.text)['inference_results'][0]['output'][0]['data'][0:10]
shape = json.loads(r.text)['inference_results'][0]['output'][0]['shape'][0]
print("First 10 dimensions:")
print(str(embed))
print("\n")
print("Total: "+str(shape)+" dimensions")


# ## 5. Create the OpenSearch ingest pipeline

# In[ ]:


path = "_ingest/pipeline/bedrock-multimodal-ingest-pipeline"
url = 'https://'+host + '/' + path
payload = {
"description": "A text/image embedding pipeline",
"processors": [
{
"text_image_embedding": {
"model_id":model_id,
"embedding": "vector_embedding",
"field_map": {
"text": "product_description",
"image": "image_binary"
}}}]}
r = requests.put(url, auth=awsauth, json=payload, headers=headers)
print(r.status_code)
print(r.text)


# ## 6. Create the k-NN index

# In[ ]:


path = "bedrock-multimodal-demostore-search-index"
url = 'https://'+host + '/' + path

#this will delete the index if already exists
requests.delete(url, auth=awsauth, json=payload, headers=headers)

payload = {
  "settings": {
    "index.knn": True,
    "default_pipeline": "bedrock-multimodal-ingest-pipeline"
  },
  "mappings": {
      
    "_source": {
     
      "excludes": ["image_binary"]
    },
    "properties": {
      "vector_embedding": {
        "type": "knn_vector",
        "dimension": shape,
        "method": {
          "name": "hnsw",
          "engine": "faiss",
          "parameters": {}
        }
      },
      "product_description": {
        "type": "text"
      },
        "image_url": {
        "type": "text"
      },
      "image_binary": {
        "type": "binary"
      }
    }
  }
}
r = requests.put(url, auth=awsauth, json=payload, headers=headers)
print(r.status_code)
print(r.text)


# ## 7. Ingest the dataset into k-NN index usig Bulk request

# In[ ]:


from ruamel.yaml import YAML
from PIL import Image
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

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

# Load the products from the dataset
yaml = YAML()
items_ = yaml.load(open('tmp/images/products.yaml'))

batch = 0
count = 0
body_ = ''
batch_size = 100
last_batch = int(len(items_)/batch_size)
action = json.dumps({ 'index': { '_index': 'bedrock-multimodal-demostore-search-index' } })

for item in items_:
    count+=1
    fileshort = "tmp/images/"+item["category"]+"/"+item["image"]
    payload = {}
    payload['image_url'] = fileshort
    payload['product_description'] = item['description']
    
    #resize the image and generate image binary
    file_type, path = resize_image(fileshort, 2048, 2048)

    with open(fileshort.split(".")[0]+"-resized."+file_type, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode("utf8")
    
    os.remove(fileshort.split(".")[0]+"-resized."+file_type)
    payload['image_binary'] = input_image
    
    body_ = body_ + action + "\n" + json.dumps(payload) + "\n"
    
    if(count == batch_size):
        response = aos_client.bulk(
        index = 'bedrock-multimodal-demostore-search-index',
        body = body_
        )
        batch += 1
        count = 0
        print("batch "+str(batch) + " ingestion done!")
        if(batch != last_batch):
            body_ = ""
        
            
#ingest the remaining rows
response = aos_client.bulk(
        index = 'bedrock-multimodal-demostore-search-index',
        body = body_
        )
        
print("All "+str(last_batch)+" batches ingested into index")


# ## 8. Experiment 1: Keyword search

# In[ ]:


#Keyword Search
query = "trendy footwear for women"
url = 'https://' + host + "/bedrock-multimodal-demostore-search-index/_search"
keyword_payload = {"_source": {
        "exclude": [
            "vector_embedding"
        ]
        },
        "query": {    "match": {
                        "product_description": {
                            "query": query
                        }
                        }
                    }
        
        ,"size":5,
  }

r = requests.get(url, auth=awsauth, json=keyword_payload, headers=headers)
response_ = json.loads(r.text)
docs = response_['hits']['hits']

for i,doc in enumerate(docs):
    print(str(i+1)+ ". "+doc["_source"]["product_description"])
    image = Image.open(doc["_source"]["image_url"])
    image.show()


# ## 9. Experiment 2: Multimodal search with only text caption as input

# In[ ]:


#Multimodal Search
#Text as input

query = "trendy footwear for women"
url = 'https://'+host+"/bedrock-multimodal-demostore-search-index/_search"
keyword_payload = {"_source": {
        "exclude": [
            "vector_embedding"
        ]
        },
        "query": {    
       
        "neural": {
            "vector_embedding": {
                
            #"query_image":query_image_binary,
            "query_text":query,
                
            "model_id": model_id,
            "k": 3
            }
            }
            
                    }
        
        ,"size":5,
  }

r = requests.get(url, auth=awsauth, json=keyword_payload, headers=headers)
response_ = json.loads(r.text)
docs = response_['hits']['hits']

for i,doc in enumerate(docs):
    print(doc["_source"]["product_description"])
    image = Image.open(doc["_source"]["image_url"])
    image.show()
   
    


# ## 10. Experiment 3: Multimodal search with only image as input

# In[ ]:


#Multimodal Search
#image as input
import urllib.request
s3 = boto3.client('s3')
url = 'https://'+host + "/bedrock-multimodal-demostore-search-index/_search"
query = "trendy black footwear for women"
print("Input text query: "+query)
# urllib.request.urlretrieve( 
#   'https://cdn.pixabay.com/photo/2014/09/03/20/15/shoes-434918_1280.jpg',"tmp/women-footwear.jpg") 
image_file = urllib.request.urlretrieve('https://aws-blogs-artifacts-public.s3.amazonaws.com/BDB-3144/women_wear.jpg', 'tmp/women-footwear-1.jpg')
img = Image.open("tmp/women-footwear-1.jpg") 
print("Input query Image:")
img.show()
with open("tmp/women-footwear-1.jpg", "rb") as image_file:
    query_image_binary = base64.b64encode(image_file.read()).decode("utf8")
keyword_payload = {"_source": {
        "exclude": [
            "vector_embedding"
        ]
        },
        "query": {    
       
        "neural": {
            "vector_embedding": {
                
            "query_image":query_image_binary,
            "model_id": model_id,
            "k": 5
            }
            
            }
                    }
        
        ,"size":5,
  }

r = requests.get(url, auth=awsauth, json=keyword_payload, headers=headers)
response_ = json.loads(r.text)
docs = response_['hits']['hits']

for i,doc in enumerate(docs):
    print(doc["_source"]["product_description"])
    image = Image.open(doc["_source"]["image_url"])
    image.show()


# ## 11. Experiment 4: Multimodal search with both image and text caption as inputs

# In[ ]:


#Multimodal Search
#Text and image as inputs
import urllib.request
s3 = boto3.client('s3')
url = 'https://'+host + "/bedrock-multimodal-demostore-search-index/_search"
query = "trendy footwear for women"
print("Input text query: "+query)
# urllib.request.urlretrieve( 
#   'https://cdn.pixabay.com/photo/2014/09/03/20/15/shoes-434918_1280.jpg',"tmp/women-footwear.jpg") 
img = Image.open("tmp/women-footwear-1.jpg") 
print("Input query Image:")
img.show()
with open("tmp/women-footwear-1.jpg", "rb") as image_file:
    query_image_binary = base64.b64encode(image_file.read()).decode("utf8")
keyword_payload = {"_source": {
        "exclude": [
            "vector_embedding"
        ]
        },
        "query": {    
       
        "neural": {
            "vector_embedding": {
                
            "query_image":query_image_binary,
            "query_text":query,
                
            "model_id": model_id,
            "k": 5
            }
            
            }
                    }
        
        ,"size":5,
  }

r = requests.get(url, auth=awsauth, json=keyword_payload, headers=headers)
response_ = json.loads(r.text)
docs = response_['hits']['hits']

for i,doc in enumerate(docs):
    print(doc["_source"]["product_description"])
    image = Image.open(doc["_source"]["image_url"])
    image.show()


# In[ ]:




