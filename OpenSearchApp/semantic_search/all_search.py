'''
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
'''

from collections import namedtuple
from datetime import datetime, timedelta
from dateutil import tz, parser
import itertools
import json
import os
import time
import uuid
import requests
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth
from datetime import datetime
import boto3


# Lambda Interval Settings (seconds)
LAMBDA_INTERVAL=60

################################################################################
# Environment

DOMAIN_ENDPOINT =   "search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com" #"search-opensearchservi-rimlzstyyeih-3zru5p2nxizobaym45e5inuayq.us-west-2.es.amazonaws.com" 
REGION = "us-east-1" #'us-west-2'#
SAGEMAKER_MODEL_ID = 'uPQAE40BnrP7K1qW-Alx' #'P_vh7YsBNIVobUP3w7RJ' #
BEDROCK_TEXT_MODEL_ID = 'iUvGQYwBuQkLO8mDfE0l'
BEDROCK_MULTIMODAL_MODEL_ID = 'o6GykYwB08ySW4fgXdf3'
SAGEMAKER_SPARSE_MODEL_ID = 'srrJ-owBQhe1aB-khx2n'

current_date_time = (datetime.now()).isoformat()
today_ = datetime.today().strftime('%Y-%m-%d')


# #session=boto3.Session(
        
# aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
# aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
# aws_session_token=os.environ['AWS_SESSION_TOKEN']
# #    )


#awsauth = AWS4Auth(aws_access_key_id, aws_secret_access_key, REGION, 'es', session_token=aws_session_token)

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, 'es', session_token=credentials.token)


# Lambda handler
def handler(input_,session_id):
    
    
  
    print("*********")
    print(input_)
    search_type = input_['searchType']
    
    if("NormType" not in input_.keys()):
        norm_type = "min_max"
    else:
        norm_type = input_["NormType"]
        
    if("CombineType" not in input_.keys()):
        combine_type = "arithmetic_mean"
    else:
        combine_type = input_["CombineType"]
    
    if("weight" not in input_.keys()):
        semantic_weight = 0.5
    else:
        semantic_weight = input_["weight"]
        
    
    search_type = input_["searchType"]
    
    if("modelType" not in input_.keys()):
        model_type = "GPT-J-6B"
    else:
        model_type = input_["modelType"]
    
    query = input_["text"]
    img = input_["image"]
    
    if("sparse" not in input_.keys()):
        sparse = "disabled"
    else:
        sparse = input_["sparse"]
    
    
    k_ = input_["K"]
    image_upload = input_["imageUpload"]
    
    if(search_type == 'Keyword Search'):
        semantic_weight = 0.0
    if(search_type == 'Vector Search'):
        semantic_weight = 1.0
    if('sagemaker' in model_type.lower()):
        index_name = 'sagemaker-search-index-retail'
        model_id = SAGEMAKER_MODEL_ID
    else:
        index_name = 'bedrock_text-search-index-retail'
        model_id = BEDROCK_TEXT_MODEL_ID
    if('Multi-modal Search' in search_type):
        index_name = 'bedrock-multimodal-demostore-search-index' # this is for retail demo store aruna's dataset
        #'bedrock-multimodal-retail-search-index' --- this is for marqo
        #'bedrock-multimodal-search-index' ----- this is for berkeley
        model_id = BEDROCK_MULTIMODAL_MODEL_ID
        
    # if(sparse == 'enabled'):
    #   index_name = 'sagemaker_sparse-search-index' #'bedrock-multimodal-search-index'
    #   model_id = SAGEMAKER_SPARSE_MODEL_ID
    
    host = 'https://'+DOMAIN_ENDPOINT+'/'
    headers = {"Content-Type": "application/json"}
    
    if('Multi-modal Search' not in search_type):
        
        
        # If search type is keyword or vector or hybrid
        
        if(sparse == 'enabled' and search_type == 'Keyword Search'):
        
        
        
        # If text expansion is enabled
        
            index_name = 'sagemaker_sparse-search-index-retail' 
            model_id = SAGEMAKER_SPARSE_MODEL_ID
            path = index_name+"/_search"
            
            
            
            url = host + path
            
            print("text expansion is enabled")
            payload = {
            # "_source": {
            #   "exclude": [
            #     "caption_embedding"
            #   ]
            # },
            "query":
                {"neural_sparse": 
                {
                "caption_embedding":
                    {
                "query_text": query,
                "model_id": model_id
                #"max_token_score": 2
            }
            }
                
                },"highlight": {
            "fields": {
            "caption": {}
            }
            }
                }
            
            r = requests.get(url, auth=awsauth, json=payload, headers=headers)
            
            
            
            print(r.status_code)
            print(r.text)
            
            path2 =  "_plugins/_ml/models/"+SAGEMAKER_SPARSE_MODEL_ID+"/_predict"
            
            url2 = host+path2
            
            payload2 = {
            "parameters": {
                "inputs": query
                }
                    }
            
            r2 = requests.post(url2, auth=awsauth, json=payload2, headers=headers)
            
            print(type(r2.text))
            
            
            sparse_ = json.loads(r2.text)
            
            print(type(sparse_))
            inter_ = json.loads(r.text)
            inter_["query_sparse"] = sparse_["inference_results"][0]["output"][0]["dataAsMap"]["response"][0]
            
            #r = {"text":json.dumps(inter_)}
            
            #r.text["query_sparse"] = r2.text["inference_results"]["output"][0]["response"][0]
            
            print(inter_)
            return json.dumps(inter_)
        else:
        
            #Create the search pipeline with normalisation processor for hybrid search to work
            
            path = "_search/pipeline/hybrid-search-pipeline" 
            url = host + path
            
            payload = {
                "description": "Post processor for hybrid search",
                "phase_results_processors": [
                {
                    "normalization-processor": {
                    "normalization": {
                        "technique": norm_type
                    },
                    "combination": {
                        "technique": combine_type,
                        "parameters": {
                        "weights": [1-semantic_weight,semantic_weight]
                        }
                    }
                    }
                }
                ]
            }
            
            r = requests.put(url, auth=awsauth, json=payload, headers=headers)
            print(r.status_code)
            
            print(r.text)
            
            
                
            
            #Create the 2 queries for hybrid search
            
            path = index_name+"/_search?search_pipeline=hybrid-search-pipeline" 
            
            url = host + path
            
            payload = {
                "_source": {
                "exclude": [
                    "caption_embedding"
                ]
                },
                "query": {
                "hybrid": {
                    "queries": [
                    
                    #############
                
                    ]
                }
                },"size":k_,
                "highlight": {
            "fields": {
            "caption": {}
            }
            }}
            
            
            #Query 1: keyword match query
            
            query_1 = {
                        "match": {
                        "caption": {
                            "query": query
                        }
                        }
                    }
                
            #Query 2: Neural query 
            
            query_2 = {
                        "neural": {
                        "caption_embedding": {
                            "query_text": query,
                            "model_id": model_id,
                            "k": k_
                        }
                        }
                    }
                    
            payload["query"]["hybrid"]["queries"].append(query_1)
            
            payload["query"]["hybrid"]["queries"].append(query_2)
            
            print(payload)     
            r = requests.get(url, auth=awsauth, json=payload, headers=headers)
            print(r.status_code)
            print(r.text)
        
    else:
        
        # This segment is for Multimodal search only
        
        
    
        path = index_name+"/_search"
        url = host + path
        
        
        payload = {
        "size": k_*2,
        "_source": {
        "exclude": [
            "image_binary","vector_embedding"
        ]
        },
        "query": {
        "neural": {
            "vector_embedding": {
            
            "model_id": model_id,
            "k": k_*2
            }
            }
            }
        }
        
        if(image_upload == 'yes' and query == ""):
            payload["query"]["neural"]["vector_embedding"]["query_image"] =  img
        if(image_upload == 'no' and query != ""):
            payload["query"]["neural"]["vector_embedding"]["query_text"] =  query
        if(image_upload == 'yes' and query != ""):
            payload["query"]["neural"]["vector_embedding"]["query_image"] =  img
            payload["query"]["neural"]["vector_embedding"]["query_text"] =  query
            
        print(payload)
        r = requests.get(url, auth=awsauth, json=payload, headers=headers)
        print(r.status_code)
        print(r.text)
        print(payload)
        

        
    response_ = json.loads(r.text)
    print(response_.keys())
    docs = response_['hits']['hits']
    #print(docs)
    arr = []
    dup = []
    if('Multi-modal Search' in search_type ):
        key_ = 'image_description'
    else:
        key_ = 'caption'
    #filter_out = 0
    
    for doc in docs:
        # if('b5/b5319e00' in doc['_source']['image_s3_url'] ):
        #     filter_out +=1
        #     continue
        
        if(doc['_source']['image_s3_url'] not in dup):
            res_ = {"desc":doc['_source'][key_],"image_url":doc['_source']['image_s3_url']}
            if('highlight' in doc):
                res_['highlight'] = doc['highlight'][key_]
            if('caption_embedding' in doc['_source']):
                #print("@@@@@@@@@@@@@@@@@@@@@")
                res_['sparse'] = doc['_source']['caption_embedding']
            if('query_sparse' in response_ and len(arr) ==0 ):
                res_['query_sparse'] = response_["query_sparse"]
            res_['id'] = doc['_id']
            res_['score'] = doc['_score']
            res_['title'] = doc['_source']['caption']
        
            arr.append(res_)
            dup.append(doc['_source']['image_s3_url'])

    size_ = input_['K']

    return arr[0:size_]
    
    

    
