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
from requests.auth import HTTPBasicAuth
from datetime import datetime
import boto3
import re
import streamlit as st




current_date_time = (datetime.now()).isoformat()
today_ = datetime.today().strftime('%Y-%m-%d')


def handler(input_,session_id):
    DOMAIN_ENDPOINT =   st.session_state.OpenSearchDomainEndpoint #"search-opensearchservi-rimlzstyyeih-3zru5p2nxizobaym45e5inuayq.us-west-2.es.amazonaws.com" 
    REGION = st.session_state.REGION
    #SAGEMAKER_MODEL_ID = st.session_state.SAGEMAKER_MODEL_ID
    BEDROCK_TEXT_MODEL_ID = st.session_state.BEDROCK_TEXT_MODEL_ID
    BEDROCK_MULTIMODAL_MODEL_ID = st.session_state.BEDROCK_MULTIMODAL_MODEL_ID
    SAGEMAKER_SPARSE_MODEL_ID = st.session_state.SAGEMAKER_SPARSE_MODEL_ID
    SAGEMAKER_CrossEncoder_MODEL_ID = st.session_state.SAGEMAKER_CrossEncoder_MODEL_ID
    profile = False
    print("BEDROCK_TEXT_MODEL_ID")
    print(BEDROCK_TEXT_MODEL_ID)
 
    ####### Auth and connection for OpenSearch domain #######
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, 'es', session_token=credentials.token)
    host = 'https://'+DOMAIN_ENDPOINT+'/'
    headers = {"Content-Type": "application/json"}
  
  
    ####### Parsing Inputs from user #######
    print("*********")
    print(input_)
    search_types = input_["searchType"]
    
    

    
    query = input_["text"]
    img = input_["image"]
    
    if("sparse" not in input_.keys()):
        sparse = "disabled"
    else:
        sparse = input_["sparse"]
    
    k_ = input_["K"]
    
    
    num_queries = len(search_types)
    
        
    ######## Inititate creating a total search pipeline #######   
    print("Creating the total pipeline")        
    s_pipeline_payload = {"version": 1234}
    
     ######## Sparse Search pipeline #######  

    path = "demostore-search-index/_search"
    
    url = host + path
    
    hybrid_payload = {
        "_source": {
        "exclude": [
            "product_description_vector","product_multimodal_vector","product_image"
        ]
        },
        "query": {
        "hybrid": {
            "queries": [
            
            #1. keyword query
            #2. vector search query
            #3. multimodal query
            #4. Sparse query
        
            ]
        }
        },"size":k_,
        "highlight": {
    "fields": {
    "product_description": {}
    }
    }}
    
    
            
    if('Keyword Search' in search_types):
        keyword_payload = {
                        "match": {
                        "product_description": {
                            "query": query
                        }
                        }
                    }
        
        
#         if(st.session_state.input_imageUpload == 'yes'):
#             keyword_payload = st.session_state.input_rewritten_query['query']
        
            
        if(st.session_state.input_manual_filter == "True"):
            keyword_payload['bool']={'filter':[]}
            if(st.session_state.input_category!=None):
                keyword_payload['bool']['filter'].append({"term": {"category": st.session_state.input_category}})
            if(st.session_state.input_gender!=None):
                keyword_payload['bool']['filter'].append({"term": {"gender_affinity": st.session_state.input_gender}})
            if(st.session_state.input_price!=(0,0)):
                keyword_payload['bool']['filter'].append({"range": {"price": {"gte": st.session_state.input_price[0],"lte": st.session_state.input_price[1] }}})
        
            keyword_payload['bool']['must'] = [{
                        "match": {
                        "product_description": {
                            "query": query
                        }
                        }
                    }]            
            del keyword_payload['match']
#         print("keyword_payload**************")   
#         print(keyword_payload)
                    
        
        hybrid_payload["query"]["hybrid"]["queries"].append(keyword_payload)
        
   
        
    if('NeuralSparse Search' in search_types):
        
        path2 =  "_plugins/_ml/models/"+SAGEMAKER_SPARSE_MODEL_ID+"/_predict"
        
        url2 = host+path2
        
        payload2 = {
        "parameters": {
            "inputs": query
            }
                }
        
        r2 = requests.post(url2, auth=awsauth, json=payload2, headers=headers)
        sparse_ = json.loads(r2.text)
        query_sparse = sparse_["inference_results"][0]["output"][0]["dataAsMap"]["response"][0]
        query_sparse_sorted = {key: value for key, 
               value in sorted(query_sparse.items(), 
                               key=lambda item: item[1],reverse=True)}
        print("text expansion is enabled")
        max_value = query_sparse_sorted[list(query_sparse_sorted.keys())[0]]
        threshold = round(max_value*st.session_state.input_sparse_filter,2)
        #print(max_value)
        query_sparse_sorted_filtered = {}
       
        rank_features = []
        for key_ in query_sparse_sorted.keys():
            if(query_sparse_sorted[key_]>=threshold):
                feature = {"rank_feature": {"field": "product_description_sparse_vector."+key_,"boost":query_sparse_sorted[key_]}}
                rank_features.append(feature)
                query_sparse_sorted_filtered[key_]=query_sparse_sorted[key_]
            else:
                break
        
        #print(query_sparse_sorted_filtered)
        #sparse_payload = {"bool":{"should":rank_features}} ## use this for vector hardcoded sparse search
        sparse_payload = {}
        sparse_payload_segment = {
        "neural_sparse": {
            "product_description_sparse_vector": {
            "query_text": query,
            "model_id": SAGEMAKER_SPARSE_MODEL_ID
            
            }
            }
            }
        
        ###### start of efficient filter applying #####
#         if(st.session_state.input_rewritten_query!=""):
#             sparse_payload['bool']['must'] = filter_['filter']['bool']['must']
            
        if(st.session_state.input_manual_filter == "True"):
            sparse_payload = {'bool':{'filter':[]}}
            if(st.session_state.input_category!=None):
                sparse_payload['bool']['filter'].append({"term": {"category": st.session_state.input_category}})
            if(st.session_state.input_gender!=None):
                sparse_payload['bool']['filter'].append({"term": {"gender_affinity": st.session_state.input_gender}})
            if(st.session_state.input_price!=(0,0)):
                sparse_payload['bool']['filter'].append({"range": {"price": {"gte": st.session_state.input_price[0],"lte": st.session_state.input_price[1] }}})
            sparse_payload['bool']['should'] = sparse_payload_segment
        else:
            sparse_payload = sparse_payload_segment
        
            
#         print("sparse_payload**************")   
#         print(sparse_payload)
            
        
        ###### end of efficient filter applying #####
        
        
        #print(sparse_payload)
            
        # sparse_payload = {
            
        #         "neural_sparse": 
        #         {
        #         "desc_embedding_sparse":
        #             {
        #         "query_text": query,
        #         "model_id": SAGEMAKER_SPARSE_MODEL_ID,
        #         #"max_token_score": 2
        #     }
        #     }
                
        #         }
        
        
        hybrid_payload["query"]["hybrid"]["queries"].append(sparse_payload)
            
            
            
            
        
   
    
    if(len(hybrid_payload["query"]["hybrid"]["queries"])==1):
        single_query = hybrid_payload["query"]["hybrid"]["queries"][0]
        del hybrid_payload["query"]["hybrid"]
        hybrid_payload["query"] = single_query
       
        
    
    
            
        ##########.###########
    if('NeuralSparse Search' in st.session_state.search_types and st.session_state.neural_sparse_two_phase_search_pipeline != ''):
        path = "demostore-search-index/_search?search_pipeline=neural_sparse_two_phase_search_pipeline" 
    url = host + path
    if(st.session_state.input_reranker!= 'None'):

        hybrid_payload["ext"] = {"rerank": {
                                      "query_context": {
                                         "query_text": query
                                      }
                                    }}
    ##########.###########
    ##########. LLM features ###########    
    print("hybrid_payload") 
    print("---------------")  
    print(hybrid_payload)
    print(url)
    r = requests.get(url, auth=awsauth, json=hybrid_payload, headers=headers)
    print(r.status_code)
    #print(r.text)
    response_ = json.loads(r.text)
            
    print("-------------------------------------------------------------------")
    print(st.session_state.input_rewritten_query)
    docs = response_['hits']['hits']
        
        
            
            
    arr = []
    dup = []
    for doc in docs:
        if(doc['_source']['image_url'] not in dup):
            res_ = {
                "desc":doc['_source']['product_description'],
               "caption":doc['_source']['caption'],
                "image_url":doc['_source']['image_url'],
               "category":doc['_source']['category'],
               "price":doc['_source']['price'],
               "gender_affinity":doc['_source']['gender_affinity'],
                "style":doc['_source']['style'],
                
                }
            if('highlight' in doc):
                res_['highlight'] = doc['highlight']['product_description']
            if('NeuralSparse Search' in search_types):
                res_['sparse'] = doc['_source']['product_description_sparse_vector']
                res_['query_sparse'] = query_sparse_sorted_filtered
#             if(st.session_state.input_rekog_label !="" or st.session_state.input_is_rewrite_query == 'enabled'):
#                 res_['rekog'] = {'color':doc['_source']['rekog_color'],'category': doc['_source']['rekog_categories'],'objects':doc['_source']['rekog_objects']}
            
            res_['id'] = doc['_id']
            res_['score'] = doc['_score']
            res_['title'] = doc['_source']['product_description']
            
        
            arr.append(res_)
            dup.append(doc['_source']['image_url'])

    #print(arr)
    return arr[0:k_]
    
    

    
