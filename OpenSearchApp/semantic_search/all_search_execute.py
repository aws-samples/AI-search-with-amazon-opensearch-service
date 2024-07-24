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
import streamlit as st


DOMAIN_ENDPOINT =   "search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com" #"search-opensearchservi-rimlzstyyeih-3zru5p2nxizobaym45e5inuayq.us-west-2.es.amazonaws.com" 
REGION = "us-east-1" #'us-west-2'#
SAGEMAKER_MODEL_ID = 'uPQAE40BnrP7K1qW-Alx' #'P_vh7YsBNIVobUP3w7RJ' #
BEDROCK_TEXT_MODEL_ID = 'iUvGQYwBuQkLO8mDfE0l'
BEDROCK_MULTIMODAL_MODEL_ID = 'o6GykYwB08ySW4fgXdf3'
SAGEMAKER_SPARSE_MODEL_ID = 'srrJ-owBQhe1aB-khx2n'

current_date_time = (datetime.now()).isoformat()
today_ = datetime.today().strftime('%Y-%m-%d')


def handler(input_,session_id):
    
    ####### Hybrid Search weights logic for throwing warning to users for inappropriate weights #######
    
    # def my_filtering_function(pair):
    #     key, value = pair
    #     if key.split("-")[0] + " Search" in st.session_state["inputs_"]["searchType"]:
    #         return True  # keep pair in the filtered dictionary
    #     else:
    #         return False  # filter pair out of the dictionary

 
    # filtered_search = dict(filter(my_filtering_function, st.session_state.input_weightage.items()))
    
    # search_types_used = ", ".join(st.session_state["inputs_"]["searchType"])
    
    # if((sum(st.session_state.weights_)!=100 or len(st.session_state["inputs_"]["searchType"])!=len(list(filter(lambda a: a >0, st.session_state.weights_)))) and len(st.session_state["inputs_"]["searchType"])!=1):
    #     st.warning('User Input Error for **WEIGHTS** :-\n\nOne or both of the below conditions was not satisfied, \n1. The total weight of all the selected search type(s): "'+search_types_used+'" should be equal to 100 \n 2. The weight of each of the search types, "'+search_types_used+'" should be greater than 0 \n\n Entered input: '+json.dumps(filtered_search)+'\n\n Please re-enter your weights to satisfy the above conditions and try again',icon = "🚨")
    #     refresh = st.button("Re-Enter")
    #     if(refresh):
    #         st.switch_page('pages/1_Semantic_Search.py')
    #     st.stop()
    
    ####### Auth and connection for OpenSearch domain #######
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, 'es', session_token=credentials.token)
    host = 'https://'+DOMAIN_ENDPOINT+'/'
    headers = {"Content-Type": "application/json"}
  
  
    ####### Parsing Inputs from user #######
    print("*********")
    print(input_)
    search_types = input_["searchType"]
    
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
        

    
    query = input_["text"]
    img = input_["image"]
    
    if("sparse" not in input_.keys()):
        sparse = "disabled"
    else:
        sparse = input_["sparse"]
    
    
    k_ = input_["K"]
    image_upload = input_["imageUpload"]
    
    path = "_search/pipeline/hybrid-search-pipeline" 
    url = host + path
    
    num_queries = len(search_types)
    
    weights = []
    
    searches = ['Keyword','Vector','Multimodal','NeuralSparse']
    for i in searches:
        weight = input_['weightage'][i+'-weight']/100
        if(weight>0.0):
            weights.append(weight)
            
      
        
    ######## Create Search pipeline #######   
    print("Creating Search pipeline")        
    s_pipeline_payload = {
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
                        "weights": weights
                        }
                    }
                    }
                }
                ]
            }
            
    r = requests.put(url, auth=awsauth, json=s_pipeline_payload, headers=headers)
    print("Search Pipeline Creation: "+str(r.status_code))
    
    ######## start of Applying LLM filters ####### 
    if(st.session_state.input_rewritten_query!=""):
            filter_ = {"filter": {
                 "bool": {
                     "must": []}}}
            filter_['filter']['bool']['must'] = st.session_state.input_rewritten_query['query']['bool']['must']
    ######## end of Applying LLM filters ####### 
    
    ######### Create the queries for hybrid search #########
    
    
    path = "retail-ml-search-index/_search?search_pipeline=hybrid-search-pipeline" 
    
    url = host + path
    
    hybrid_payload = {
        "_source": {
        "exclude": [
            "desc_embedding_bedrock-multimodal","desc_embedding_bedrock-text"
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
    "description": {}
    }
    }}
    
    
            
    if('Keyword Search' in search_types):
        
        keyword_payload = {
                        "match": {
                        "description": {
                            "query": query
                        }
                        }
                    }
        if(st.session_state.input_rewritten_query !=""):
            keyword_payload = st.session_state.input_rewritten_query['query']
        
        
        hybrid_payload["query"]["hybrid"]["queries"].append(keyword_payload)
        
    if('Vector Search' in search_types):
        
        path3 =  "_plugins/_ml/models/"+BEDROCK_TEXT_MODEL_ID+"/_predict"
        
        url3 = host+path3
        
        payload3 = {
        "parameters": {
            "inputs": query
            }
                }
        
        r3 = requests.post(url3, auth=awsauth, json=payload3, headers=headers)
        vector_ = json.loads(r3.text)
        
        query_vector = vector_['inference_results'][0]['output'][0]['data']
        #print(query_vector)
        
        vector_payload = {
                        "knn": {
                        "desc_embedding_bedrock-text": {
                            "vector":query_vector,
                            #"query_text": query,
                            #"model_id": BEDROCK_TEXT_MODEL_ID,
                            "k": k_
                        }
                        }
                    }
        
        # using neural query (without efficient filters)
        # vector_payload = {
        #                 "neural": {
        #                 "desc_embedding_bedrock-text": {
        #                     "query_text": query,
        #                     "model_id": BEDROCK_TEXT_MODEL_ID,
        #                     "k": k_
        #                 }
        #                 }
        #             }
        
        ###### start of efficient filter applying #####
        if(st.session_state.input_rewritten_query!=""):
            vector_payload['knn']['desc_embedding_bedrock-text']['filter'] = filter_['filter']
        ###### end of efficient filter applying #####
        
        hybrid_payload["query"]["hybrid"]["queries"].append(vector_payload)
        
    if('Multimodal Search' in search_types):
        
        multimodal_payload  = {
       
        "neural": {
            "desc_embedding_bedrock-multimodal": {
            
            "model_id": BEDROCK_MULTIMODAL_MODEL_ID,
            "k": k_
            }
            }
            }
        
        
        if(image_upload == 'yes' and query == ""):
            multimodal_payload["neural"]["desc_embedding_bedrock-multimodal"]["query_image"] =  img
        if(image_upload == 'no' and query != ""):
            multimodal_payload["neural"]["desc_embedding_bedrock-multimodal"]["query_text"] =  query
        if(image_upload == 'yes' and query != ""):
            
            multimodal_payload["neural"]["desc_embedding_bedrock-multimodal"]["query_image"] =  img
            multimodal_payload["neural"]["desc_embedding_bedrock-multimodal"]["query_text"] =  query
        
        
        
        hybrid_payload["query"]["hybrid"]["queries"].append(multimodal_payload)
          
          
          
        
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
        #print(query_sparse_sorted)
        query_sparse_sorted_filtered = {}
       
        rank_features = []
        for key_ in query_sparse_sorted.keys():
            if(query_sparse_sorted[key_]>=st.session_state.input_sparse_filter):
                feature = {"rank_feature": {"field": "desc_embedding_sparse."+key_,"boost":query_sparse_sorted[key_]}}
                rank_features.append(feature)
                query_sparse_sorted_filtered[key_]=query_sparse_sorted[key_]
            else:
                break
        
        #print(query_sparse_sorted_filtered)
        sparse_payload = {"bool":{"should":rank_features}}
        
        ###### start of efficient filter applying #####
        if(st.session_state.input_rewritten_query!=""):
            sparse_payload['bool']['must'] = filter_['filter']['bool']['must']
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
            
            
            
            
        
        

        
    print("hybrid_payload") 
    print("---------------")  
    docs = []
    
    if(st.session_state.input_sql_query!=""):
        url = host +"_plugins/_sql?format=json"
        payload = {"query":st.session_state.input_sql_query}
        r = requests.post(url, auth=awsauth, json=payload, headers=headers)
        print("^^^^^")
        print(r.text)
    
    if(len(hybrid_payload["query"]["hybrid"]["queries"])==1):
        single_query = hybrid_payload["query"]["hybrid"]["queries"][0]
        del hybrid_payload["query"]["hybrid"]
        hybrid_payload["query"] = single_query
        # print("-------final query--------")
        print(hybrid_payload)
        r = requests.get(url, auth=awsauth, json=hybrid_payload, headers=headers)
        print(r.status_code)
        #print(r.text)
        response_ = json.loads(r.text)
        print("-------------------------------------------------------------------")
        print(response_)
        docs = response_['hits']['hits']
    
    
    else:
        
        if( st.session_state.input_hybridType == "OpenSearch Hybrid Query"):
            r = requests.get(url, auth=awsauth, json=hybrid_payload, headers=headers)
            print(r.status_code)
            response_ = json.loads(r.text)
            print("-------------------------------------------------------------------")
            #print(response_)
            docs = response_['hits']['hits']
        
        else:
            all_docs = []
            all_docs_ids = []
            only_hits = []
        
            rrf_hits = []
            for i,query in enumerate(hybrid_payload["query"]["hybrid"]["queries"]):
                payload_ =  {'_source': 
                    {'exclude': ['desc_embedding_bedrock-multimodal', 'desc_embedding_bedrock-text']}, 
                    'query': query, 
                    'size': k_, 'highlight': {'fields': {'description': {}}}}
                
                r_ = requests.get(url, auth=awsauth, json=payload_, headers=headers)
                resp = json.loads(r_.text)
                all_docs.append({"search":list(query.keys())[0],"results":resp['hits']['hits'],"weight":weights[i]})
                only_hits.append(resp['hits']['hits'])
                for hit in resp['hits']['hits']:
                    all_docs_ids.append(hit["_id"])
                    
            
            id_scores = []
            rrf_hits_unsorted = []
            
            for id in all_docs_ids:
                score = 0.0
                for result_set in all_docs:
                    if id in json.dumps(result_set['results']):
                        for n,res in enumerate(result_set['results']):
                            if(res["_id"] == id):
                                score += result_set["weight"] * (1.0 /  (n+1))
                id_scores.append({"id":id,"score":score})
                for only_hit in only_hits:
                    for i_ in only_hit:
                        if(id == i_["_id"]):
                            i_["_score"] = score
                        rrf_hits_unsorted.append(i_)
            print("rrf_hits_unsorted------------------------------")   
            docs = sorted(rrf_hits_unsorted, key=lambda x: x['_score'],reverse=True)
            #print(docs)
        
        
            
            
    arr = []
    dup = []
    for doc in docs:
        if(doc['_source']['image_s3_url'] not in dup):
            res_ = {
                "desc":doc['_source']['description'],
                "caption":doc['_source']['caption'],
                "image_url":doc['_source']['image_s3_url'],
                "category":doc['_source']['category'],
                "price":doc['_source']['price'],
                "gender_affinity":doc['_source']['gender_affinity'],
                "style":doc['_source']['style'],
                
                }
            if('highlight' in doc):
                res_['highlight'] = doc['highlight']['description']
            if('NeuralSparse Search' in search_types):
                res_['sparse'] = doc['_source']['desc_embedding_sparse']
                res_['query_sparse'] = query_sparse_sorted_filtered
            if(st.session_state.input_rekog_label !="" or st.session_state.input_is_rewrite_query == 'enabled'):
                res_['rekog'] = {'color':doc['_source']['rekog_color'],'category': doc['_source']['rekog_categories'],'objects':doc['_source']['rekog_objects']}
            
            res_['id'] = doc['_id']
            res_['score'] = doc['_score']
            res_['title'] = doc['_source']['description']
            
        
            arr.append(res_)
            dup.append(doc['_source']['image_s3_url'])


    return arr[0:k_]
    
    

    
