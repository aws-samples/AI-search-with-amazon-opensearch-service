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
    
    
    
    num_queries = len(search_types)
    
    weights = []
    
    searches = ['Keyword','Vector','Multimodal','NeuralSparse']
    for i in searches:
        weight = input_['weightage'][i+'-weight']/100
        if(weight>0.0):
            weights.append(weight)
            
      
        
    ######## Inititate creating a total search pipeline #######   
    print("Creating the total pipeline")        
    s_pipeline_payload = {"version": 1234}
    
     ######## Sparse Search pipeline #######  
    if('NeuralSparse Search' in st.session_state.search_types and st.session_state.neural_sparse_two_phase_search_pipeline != ''):
        path = "_search/pipeline/neural_sparse_two_phase_search_pipeline" 
        sparse_pipeline_payload = (json.loads(st.session_state.neural_sparse_two_phase_search_pipeline))['neural_sparse_two_phase_search_pipeline']
        #updating prune
        sparse_pipeline_payload['request_processors'][0]['neural_sparse_two_phase_processor']['two_phase_parameter']['prune_ratio'] = st.session_state.input_sparse_filter
        url = host + path
        r = requests.put(url, auth=awsauth, json=sparse_pipeline_payload, headers=headers)
        print("Sparse Search Pipeline updated: "+str(r.status_code))
        s_pipeline_payload['request_processors'] = sparse_pipeline_payload['request_processors']
    
    
    ######## Hybrid Search pipeline #######  
    opensearch_search_pipeline = (requests.get(host+'_search/pipeline/hybrid_search_pipeline', auth=awsauth,headers=headers)).text
    print("opensearch_search_pipeline")
    print(opensearch_search_pipeline)
    
    if(opensearch_search_pipeline!='{}'):
        path = "_search/pipeline/hybrid_search_pipeline" 
        url = host + path
        hybrid_pipeline_payload = {"phase_results_processors":[
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
                ]}
        r = requests.put(url, auth=awsauth, json=hybrid_pipeline_payload, headers=headers)
        print("Hybrid Search Pipeline updated: "+str(r.status_code))
        s_pipeline_payload['phase_results_processors'] = hybrid_pipeline_payload['phase_results_processors']
        
     ######## Rerank pipeline #######     
    if(st.session_state.input_reranker!= 'None'):
        opensearch_sagemaker_rerank_pipeline = (requests.get(host+'_search/pipeline/sagemaker_rerank_pipeline', auth=awsauth,headers=headers)).text
        print("opensearch_sagemaker_rerank_pipeline")
        print(opensearch_sagemaker_rerank_pipeline)
        opensearch_bedrock_rerank_pipeline = (requests.get(host+'_search/pipeline/bedrock_rerank_pipeline', auth=awsauth,headers=headers)).text
        print("opensearch_bedrock_rerank_pipeline")
        print(opensearch_bedrock_rerank_pipeline)
        if(opensearch_sagemaker_rerank_pipeline!='{}' or opensearch_bedrock_rerank_pipeline!='{}'):
            if(opensearch_sagemaker_rerank_pipeline!='{}' and st.session_state.input_reranker == 'SageMaker Cross Encoder'):
                total_pipeline = json.loads(opensearch_sagemaker_rerank_pipeline)
                sagemaker_or_bedrock = 'sagemaker_rerank_pipeline'
            if(opensearch_bedrock_rerank_pipeline!='{}' and st.session_state.input_reranker == 'Bedrock Rerank'):
                total_pipeline = json.loads(opensearch_bedrock_rerank_pipeline)
                sagemaker_or_bedrock = 'bedrock_rerank_pipeline'
            s_pipeline_payload['response_processors'] = total_pipeline[sagemaker_or_bedrock]['response_processors']

    ######## Combining hybrid+rerank+sparse pipeline ####### 
    if(len(list(s_pipeline_payload.keys()))>1):
        path = "_search/pipeline/hybrid_rerank_sparse_pipeline" 
        url = host + path
        r = requests.put(url, auth=awsauth, json=s_pipeline_payload, headers=headers)
        print("hybrid_rerank_sparse_pipeline Creation: "+str(r.status_code))


    if(st.session_state.input_is_rewrite_query == 'enabled' and st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType and st.session_state.llm_search_request_pipeline_image == "true"):    
        llm_text = json.loads((requests.get(host+'_search/pipeline/LLM_search_request_pipeline', auth=awsauth,headers=headers)).text)
        llm_image = json.loads((requests.get(host+'_search/pipeline/LLM_search_request_pipeline_image', auth=awsauth,headers=headers)).text)
        new_desc = "combined LLM text and image pipeline"
        new_processors = llm_image['LLM_search_request_pipeline_image']['request_processors']
        new_processors.append(llm_text['LLM_search_request_pipeline']['request_processors'][0])
        new_payload = {"description":new_desc,"request_processors":new_processors}
        path = "_search/pipeline/LLM_search_request_pipeline_text_image" 
        url = host + path
        r = requests.put(url, auth=awsauth, json=new_payload, headers=headers)
        print("combined LLM text and image pipeline Creation: "+str(r.status_code))
        
    
    ######## start of Applying LLM filters ####### 
#     if(st.session_state.input_is_rewrite_query == 'enabled'):
#         if(st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType):
#             filter_ = {"filter": {
#                  "bool": {
#                      "must": []}}}
#             filter_['filter']['bool']['must'] = st.session_state.input_rewritten_query['query']['bool']['must']
#     ######## end of Applying LLM filters ####### 
    
    ######### Create the queries for hybrid search #########
    
    
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
        if(st.session_state.input_is_rewrite_query == 'enabled' or st.session_state.input_imageUpload == 'yes'):
            profile = True
            keyword_payload = {  
                   "bool":{  
                            "must":[
                                  {
                                    "match": {"product_description": {"query":""}}
                                  }
                                    ]
                         
                          }}
            if(st.session_state.input_is_rewrite_query == 'enabled'):
                keyword_payload['bool']["must"][0]["match"]["product_description"]["query"] = query
                keyword_payload['bool']["should"]=[
                                  {
                                     "multi_match" : {
                                            "query": "",
                                            "type": "cross_fields",
                                            "fields": [ "gender_affinity","category"],
                                            "operator": "and"}
                                                      }
                                                    ]

            if(st.session_state.input_imageUpload == 'yes'):
                keyword_payload['bool']["must"][0]["match"]["product_description"]["query"] = st.session_state.input_image
                
            
               
            
        else:
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
        
    if('Vector Search' in search_types):
        
#         path3 =  "_plugins/_ml/models/"+BEDROCK_TEXT_MODEL_ID+"/_predict"
        
#         url3 = host+path3
        
#         payload3 = {
#         "parameters": {
#             "inputText": query
#             }
#                 }
        
#         r3 = requests.post(url3, auth=awsauth, json=payload3, headers=headers)
#         vector_ = json.loads(r3.text)
#         #print(r3.text)
#         query_vector = vector_['inference_results'][0]['output'][0]['data']
#         #print(query_vector)
        
#         vector_payload = {
#                         "knn": {
#                         "product_description_vector": {
#                             "vector":query_vector,
#                             #"query_text": query,
#                             #"model_id": BEDROCK_TEXT_MODEL_ID,
#                             "k": k_
#                         }
#                         }
#                     }
        
        #using neural query 
        vector_payload = {
                        "neural": {
                        "product_description_vector": {
                            "query_text": query,
                            "model_id": BEDROCK_TEXT_MODEL_ID,
                            "k": k_
                        }
                        }
                    }
        
        ###### start of efficient filter applying #####
#         if(st.session_state.input_rewritten_query!=""):
#             vector_payload['neural']['product_description_vector']['filter'] = filter_['filter']
            
        if(st.session_state.input_manual_filter == "True"):
            vector_payload['neural']['product_description_vector']['filter'] = {"bool":{"must":[]}}
            if(st.session_state.input_category!=None):
                vector_payload['neural']['product_description_vector']['filter']["bool"]["must"].append({"term": {"category": st.session_state.input_category}})
            if(st.session_state.input_gender!=None):
                vector_payload['neural']['product_description_vector']['filter']["bool"]["must"].append({"term": {"gender_affinity": st.session_state.input_gender}})
            if(st.session_state.input_price!=(0,0)):
                vector_payload['neural']['product_description_vector']['filter']["bool"]["must"].append({"range": {"price": {"gte": st.session_state.input_price[0],"lte": st.session_state.input_price[1] }}})
        
#         print("vector_payload**************")   
#         print(vector_payload)    
        
        ###### end of efficient filter applying #####
        
        hybrid_payload["query"]["hybrid"]["queries"].append(vector_payload)
        
    if('Multimodal Search' in search_types):
        
        multimodal_payload  = {
       
        "neural": {
            "product_multimodal_vector": {
            
            "model_id": BEDROCK_MULTIMODAL_MODEL_ID,
            "k": k_
            }
            }
            }
        
        
        if(image_upload == 'yes' and query == ""):
            multimodal_payload["neural"]["product_multimodal_vector"]["query_image"] =  img
        if(image_upload == 'no' and query != ""):
            multimodal_payload["neural"]["product_multimodal_vector"]["query_text"] =  query
        if(image_upload == 'yes' and query != ""):
            
            multimodal_payload["neural"]["product_multimodal_vector"]["query_image"] =  img
            multimodal_payload["neural"]["product_multimodal_vector"]["query_text"] =  query
        
        ###### start of efficient filter applying #####
#         if(st.session_state.input_rewritten_query!=""):
#             multimodal_payload['neural']['product_multimodal_vector']['filter'] = filter_['filter']
            
        if(st.session_state.input_manual_filter == "True"):
            print("presence of filters------------")
            multimodal_payload['neural']['product_multimodal_vector']['filter'] = {"bool":{"must":[]}}
            if(st.session_state.input_category!=None):
                multimodal_payload['neural']['product_multimodal_vector']['filter']["bool"]["must"].append({"term": {"category": st.session_state.input_category}})
            if(st.session_state.input_gender!=None):
                multimodal_payload['neural']['product_multimodal_vector']['filter']["bool"]["must"].append({"term": {"gender_affinity": st.session_state.input_gender}})
            if(st.session_state.input_price!=(0,0)):
                multimodal_payload['neural']['product_multimodal_vector']['filter']["bool"]["must"].append({"range": {"price": {"gte": st.session_state.input_price[0],"lte": st.session_state.input_price[1] }}})
        
#         print("vector_payload**************")   
#         print(vector_payload)    
        
        ###### end of efficient filter applying #####
        
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
            
            
            
            
        
        

        
    
    print(st.session_state.bedrock_re_ranker)
    print(st.session_state.input_reranker)
    
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
        ##########. LLM features ###########
        if(st.session_state.input_is_rewrite_query == 'enabled'):
            path = "demostore-search-index/_search?search_pipeline=LLM_search_request_pipeline" 
        if(st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType and st.session_state.llm_search_request_pipeline_image == "true"):    
            path = "demostore-search-index/_search?search_pipeline=LLM_search_request_pipeline_image"
            
        if(st.session_state.input_is_rewrite_query == 'enabled' and st.session_state.input_imageUpload == 'yes' and 'Keyword Search' in st.session_state.input_searchType and st.session_state.llm_search_request_pipeline_image == "true"):
            path = "demostore-search-index/_search?search_pipeline=LLM_search_request_pipeline_text_image"
        url = host + path
            
        
        if(profile):
            hybrid_payload['profile'] = "true" 
        ##########. LLM features ###########
        
    
    
    elif(1!=1):# use for RRF
        all_docs = []
        all_docs_ids = []
        only_hits = []

        rrf_hits = []
        for i,query in enumerate(hybrid_payload["query"]["hybrid"]["queries"]):
            payload_ =  {'_source': 
                {'exclude': ['desc_embedding_bedrock-multimodal', 'desc_embedding_bedrock-text', 'product_description_sparse_vector']}, 
                'query': query, 
                'size': k_, 'highlight': {'fields': {'product_description': {}}}}

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
            
        ##########.###########
    if(len(list(s_pipeline_payload.keys()))>1 and profile==False):
        path = "demostore-search-index/_search?search_pipeline=hybrid_rerank_sparse_pipeline" 
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
    st.session_state.input_rewritten_query = ""
    if('profile' in hybrid_payload):
        split_arr = response_['profile']['shards'][0]['searches'][0]['query'][0]['description'].split("product_description:")
        print(split_arr)
        if(st.session_state.input_is_rewrite_query == 'enabled'):
            x = split_arr[-1].split(" (")[1]
            filters = ' '.join(set((' '.join(re.sub(r'[^a-zA-Z0-9]', ' ', x.replace("category","").replace("gender_affinity","")).split())).split(" ")))
            hybrid_payload['query']['bool']["should"][0]["multi_match"]["query"] = filters
            st.session_state.input_rewritten_query = hybrid_payload['query']
        if(st.session_state.input_imageUpload == 'yes'):
            split_arr[-1] = split_arr[-1].split(" (")[0]
            trans_query = re.sub(r"[^a-zA-Z0-9]+", ' ',''.join(split_arr)).strip()
            hybrid_payload['query']['bool']["must"][0]["match"]["product_description"]["query"] = trans_query
            st.session_state.input_text = trans_query
            st.session_state.input_rewritten_query = hybrid_payload['query']

      ##########.###########
            
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
    
    

    
