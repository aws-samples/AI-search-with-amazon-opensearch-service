import boto3
import json
import os
import shutil
from unstructured.partition.pdf import partition_pdf
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import streamlit as st
from PIL import Image 
import base64
import re
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import base64
from anthropic import Anthropic
import requests 
import utilities.re_ranker as re_ranker
import utilities.invoke_models as invoke_models
#import langchain
headers = {"Content-Type": "application/json"}
host = "https://search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com/"

parent_dirname = "/".join((os.path.dirname(__file__)).split("/")[0:-1])

def query_(awsauth,inputs, session_id,search_types):



    question = inputs['query']
    
    k=1
    embedding = invoke_models.invoke_model_mm(question,"none")
    
    query_mm = {
        "size": k,
          "_source": {
        "exclude": [
            "processed_element_embedding_bedrock-multimodal","processed_element_embedding_sparse","image_encoding","processed_element_embedding"
        ]
        },
        "query":  {
            "knn": {
                "processed_element_embedding_bedrock-multimodal": {
                    "vector": embedding, 
                    "k": k}
                }
        }
    }

    path = st.session_state.input_index+"_mm/_search"
    url = host+path
    r = requests.get(url, auth=awsauth, json=query_mm, headers=headers)
    response_mm = json.loads(r.text)
    # response_mm = ospy_client.search(
    #     body = query_mm,
    #     index = st.session_state.input_index+"_mm"
    # )

    

    hits = response_mm['hits']['hits']
    context = []
    context_tables = []
    images = []

    for hit in hits:
        #context.append(hit['_source']['caption'])
        images.append({'file':hit['_source']['image'],'caption':hit['_source']['processed_element']})
    
    ####### SEARCH ########
    
    
    path = "_search/pipeline/rag-search-pipeline" 
    url = host + path
    
    num_queries = len(search_types)
    
    weights = []
    
    searches = ['Keyword','Vector','NeuralSparse']
    equal_weight = (int(100/num_queries) )/100
    if(num_queries>1):
        for index,search in enumerate(search_types):
            
            if(index != (num_queries-1)):
                weight = equal_weight
            else:
                weight = 1-sum(weights)
                
            weights.append(weight)
        
        #print(weights) 
        
            
        s_pipeline_payload = {
                "description": "Post processor for hybrid search",
                "phase_results_processors": [
                {
                    "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                        "weights": weights
                        }
                    }
                    }
                }
                ]
            }
            
        r = requests.put(url, auth=awsauth, json=s_pipeline_payload, headers=headers)
        #print(r.status_code)
        #print(r.text)
        
    
    
    SIZE = 5
    
    hybrid_payload = {
        "_source": {
        "exclude": [
            "processed_element_embedding","processed_element_embedding_sparse"
        ]
        },
        "query": {
        "hybrid": {
            "queries": [
            
            #1. keyword query
            #2. vector search query
            #3. Sparse query
        
            ]
        }
        },"size":SIZE,
    }
    
    
            
    if('Keyword Search' in search_types):
        
        keyword_payload = {
                        "match": {
                        "processed_element": {
                            "query": question
                        }
                        }
                    }
        
        hybrid_payload["query"]["hybrid"]["queries"].append(keyword_payload)
        
    
        
    if('Vector Search' in search_types):
        
        embedding = embedding = invoke_models.invoke_model(question)
        
        vector_payload = {
            "knn": {
                 "processed_element_embedding": {
                     "vector": embedding, 
                     "k": 2}
                 }
                        }
                
        hybrid_payload["query"]["hybrid"]["queries"].append(vector_payload)
        
    if('Sparse Search' in search_types):
            
        #print("text expansion is enabled")
        sparse_payload =  {  "neural_sparse": {
                "processed_element_embedding_sparse": {
                    "query_text": question,
                    "model_id": "srrJ-owBQhe1aB-khx2n"
                }
                }}
                    
        
        hybrid_payload["query"]["hybrid"]["queries"].append(sparse_payload)
        
        # path2 =  "_plugins/_ml/models/srrJ-owBQhe1aB-khx2n/_predict"
        # url2 = host+path2
        # payload2 = {
        # "parameters": {
        #     "inputs": question
        #     }
        #         }
        # r2 = requests.post(url2, auth=awsauth, json=payload2, headers=headers)
        # sparse_ = json.loads(r2.text)
        # query_sparse = sparse_["inference_results"][0]["output"][0]["dataAsMap"]["response"][0]
        
   
        

        
    # print("hybrid_payload") 
    # print("---------------") 
    #print(hybrid_payload) 
    hits = []
    if(num_queries>1):
        path = st.session_state.input_index+"/_search?search_pipeline=rag-search-pipeline" 
    else:
        path = st.session_state.input_index+"/_search"
    url = host+path
    if(len(hybrid_payload["query"]["hybrid"]["queries"])==1):
        single_query = hybrid_payload["query"]["hybrid"]["queries"][0]
        del hybrid_payload["query"]["hybrid"]
        hybrid_payload["query"] = single_query
        r = requests.get(url, auth=awsauth, json=hybrid_payload, headers=headers)
        #print(r.status_code)
        response_ = json.loads(r.text)
        #print("-------------------------------------------------------------------")
        #print(r.text)
        hits = response_['hits']['hits']
        
    else:
        r = requests.get(url, auth=awsauth, json=hybrid_payload, headers=headers)
        #print(r.status_code)
        response_ = json.loads(r.text)
        #print("-------------------------------------------------------------------")
        #print(response_)
        hits = response_['hits']['hits']
    
    ##### GET reference tables separately like *_mm index search for images  ######
    def lazy_get_table():
        #print("Forcing table analysis")
        table_ref = []
        any_table_exists = False
        for fname in os.listdir(parent_dirname+"/split_pdf_csv"):
            if fname.startswith(st.session_state.input_index):
                any_table_exists = True
                break       
        if(any_table_exists):
            #################### Basic Match query #################
            # payload_tables = {
            #                     "query": {
            #                         "bool":{
                                
            #                         "must":{"match": {
            #                                         "processed_element": question
                                                
            #                                     }},
                                                
            #                             "filter":{"term":{"raw_element_type": "table"}}
                                    
                                
            #                     }}}
            
            #################### Neural Sparse query #################
            payload_tables = {"query":{"neural_sparse": {
                    "processed_element_embedding_sparse": {
                        "query_text": question,
                        "model_id": "srrJ-owBQhe1aB-khx2n"
                    }
                    }  }     }
            
            
            r_ = requests.get(url, auth=awsauth, json=payload_tables, headers=headers)
            r_tables = json.loads(r_.text)
            
            for res_ in r_tables['hits']['hits']:
                if(res_["_source"]['raw_element_type'] == 'table'):
                    table_ref.append({'name':res_["_source"]['table'],'text':res_["_source"]['processed_element']})
                if(len(table_ref) == 2):
                    break
                    
            
        return table_ref
        
        
    ########################### LLM Generation ########################
    prompt_template = """
        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context.
        {context}
        Instruction: Based on the above documents, provide a detailed answer for, {question}. Answer "don't know", 
        if not present in the context. 
        Solution:"""
        
    
    
    idx = 0
    images_2 = []
    is_table_in_result = False
    df = []
    for hit in hits[0:3]:
        
        
        if(hit["_source"]["raw_element_type"] == 'table'):
            #print("Need to analyse table")
            is_table_in_result = True
            table_res = invoke_models.read_from_table(hit["_source"]["table"],question)
            df.append({'name':hit["_source"]["table"],'text':hit["_source"]["processed_element"]})
            context_tables.append(table_res+"\n\n"+hit["_source"]["processed_element"])
            
        else:
            if(hit["_source"]["image"]!="None"):
                with open(parent_dirname+'/figures/'+st.session_state.input_index+"/"+hit["_source"]["raw_element_type"].split("_")[1].replace(".jpg","")+"-resized.jpg", "rb") as read_img:
                    input_encoded = base64.b64encode(read_img.read()).decode("utf8")
                context.append(invoke_models.generate_image_captions_llm(input_encoded,question))
            else:
                context.append(hit["_source"]["processed_element"])
            
        if(hit["_source"]["image"]!="None"):
            images_2.append({'file':hit["_source"]["image"],'caption':hit["_source"]["processed_element"]})
            
        idx = idx +1
        #images.append(hit['_source']['image'])
    
    # if(is_table_in_result == False):
    #     df = lazy_get_table()
    #     print("forcefully selected top 2 tables")
    #     print(df)
        
    #     for pos,table in enumerate(df):
    #         table_res = invoke_models.read_from_table(table['name'],question)
    #         context_tables.append(table_res)#+"\n\n"+table['text']
    
    
    total_context = context_tables + context
    
    ####### Re-Rank ########
    
    #print("re-rank")
    
    if(st.session_state.input_is_rerank == True and len(total_context)):
        ques = [{"question":question}]
        ans = [{"answer":total_context}]
        
        total_context = re_ranker.re_rank('rag','Cross Encoder',"",ques, ans)
        
    # print("-------------")
    # print("text_context")
    # print(context)
    # print("-------------")
    # print("-------------")
    # print("table_context")
    # print(context_tables)
    # print("-------------")
    # print("-------------")
    # print(total_context)
    # print("-------------")

    llm_prompt = prompt_template.format(context=total_context[0],question=question)
    output = invoke_models.invoke_llm_model( "\n\nHuman: {input}\n\nAssistant:".format(input=llm_prompt) ,False)
    #print(output)
    if(len(images_2)==0):
        images_2 = images
    return {'text':output,'source':total_context,'image':images_2,'table':df}
