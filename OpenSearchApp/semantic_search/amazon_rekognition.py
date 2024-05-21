import json
import os
import sys
import boto3
from botocore.config import Config
import getpass
import os
import streamlit as st


from opensearchpy import OpenSearch, RequestsHttpConnection
aos_host = 'search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com'

auth = ("username","password") #### input credentials

aos_client = OpenSearch(
    hosts = [{'host': aos_host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)
rekog_client = boto3.client('rekognition', region_name='us-east-1')

def extract_image_metadata(img):
    res = rekog_client.detect_labels(
    Features= [ "GENERAL_LABELS","IMAGE_PROPERTIES" ],
    Image = { 
        
        "Bytes":img
    },
    MaxLabels = 10,
    MinConfidence = 80.0,
    Settings = { 
    #       "GeneralLabels": { 
    #          "LabelCategoryExclusionFilters": [ "string" ],
    #          "LabelCategoryInclusionFilters": [ "string" ],
    #          "LabelExclusionFilters": [ "string" ],
    #          "LabelInclusionFilters": [ "string" ]
    #       },
        "ImageProperties": { 
            "MaxDominantColors": 5
        }
    }
    )
    objects_category_color = {}
    objects_category_color_masked = {}
    
    def add_span(x,type):
        if(type == 'obj'):
            return "<span style='color:#e28743;font-weight:bold'>"+x+"</span>"
        if(type == 'cat'):
            return "<span style='color:green;font-style:italic'>"+x+"</span>"
        if(type == 'color'):
            return "<span style='color:"+x+";font-weight:bold'>"+x+"</span>"
        
    
    for label in res['Labels']:
        objects_category_color_masked[add_span(label['Name'],'obj')]={'categories':[],'color':""}
        objects_category_color[label['Name']] = {'categories':[],'color':""}
        if(len(label['Categories'])!=0):
            for category in label['Categories']:
                objects_category_color_masked[add_span(label['Name'],'obj')]['categories'].append(add_span(category['Name'].lower(),'cat'))
                objects_category_color[label['Name']]['categories'].append(category['Name'].lower())


        if(len(label['Instances'])!=0):
            for instance in label['Instances']:
                if(len(instance['DominantColors'])!=0):
                    objects_category_color[label['Name']]['color'] = instance['DominantColors'][0]['SimplifiedColor']
                    objects_category_color_masked["<span style='color:#e28743;font-weight:bold'>"+label['Name']+"</span>"]['color'] = add_span(instance['DominantColors'][0]['SimplifiedColor'],'color')
                
        st.session_state.input_rekog_directoutput = objects_category_color_masked
    objects = []
    categories = []
    colors = []
    for key in objects_category_color.keys():
        if(key.lower() not in objects):
            objects.append(key.lower())
        categories.append(" ".join(set(objects_category_color[key]['categories'])))
        if(objects_category_color[key]['color']!=''):
            colors.append(objects_category_color[key]['color'].lower())
            
    objects = " ".join(set(objects))
    categories = " ".join(set(categories))
    colors = " ".join(set(colors))
    
    print("^^^^^^^^^^^^^^^^^^")
    print(colors+ " " + objects + " " + categories)
    
    return colors+ " " + objects + " " + categories

def call(a,b):
    print("'''''''''''''''''''''''")
    print(b)
    
    if(st.session_state.input_is_rewrite_query == 'enabled' and st.session_state.input_rewritten_query!=""):
        
        
        st.session_state.input_rewritten_query['query']['bool']['should'].pop()
        st.session_state.input_rewritten_query['query']['bool']['should'].append( {
                    "simple_query_string": {
                    
                        "query": a + " " + b,
                        "fields":['description','rekog_all^3']
                    
                    }
                })
        rekog_query = st.session_state.input_rewritten_query
        
    else:
        rekog_query = { "query":{
                "simple_query_string": {
                  
                    "query": a + " " + b,
                    "fields":['description','rekog_all^3']
                  
                }
              }
            }
        st.session_state.input_rewritten_query = rekog_query
        
    # response = aos_client.search(
    #     body = rekog_query,
    #     index = 'demo-retail-rekognition'
    #     #pipeline = 'RAG-Search-Pipeline'
    # )
    
    
    # hits = response['hits']['hits']
    # print("rewrite-------------------------")
    # arr = []
    # for doc in hits:
    #     # if('b5/b5319e00' in doc['_source']['image_s3_url'] ):
    #     #     filter_out +=1
    #     #     continue
        
    #     res_ = {"desc":doc['_source']['text'].replace(doc['_source']['metadata']['rekog_all']," ^^^ " +doc['_source']['metadata']['rekog_all']),
    #             "image_url":doc['_source']['metadata']['image_s3_url']}
    #     if('highlight' in doc):
    #         res_['highlight'] = doc['highlight']['text']
    #     # if('caption_embedding' in doc['_source']):
    #     #     res_['sparse'] = doc['_source']['caption_embedding']
    #     # if('query_sparse' in response_ and len(arr) ==0 ):
    #     #     res_['query_sparse'] = response_["query_sparse"]
    #     res_['id'] = doc['_id']
    #     res_['score'] = doc['_score']
    #     res_['title'] = doc['_source']['text']
    #     res_['rekog'] = {'color':doc['_source']['metadata']['rekog_color'],'category': doc['_source']['metadata']['rekog_categories'],'objects':doc['_source']['metadata']['rekog_objects']}
           
    #     arr.append(res_)
            

    
    # return arr