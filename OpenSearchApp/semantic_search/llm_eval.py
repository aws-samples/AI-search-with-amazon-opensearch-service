import os
import io
import sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2])+"/utilities")
import json
import glob
import boto3
import base64
import logging
import requests
import numpy as np
import pandas as pd
from PIL import Image
from typing import List
from botocore.auth import SigV4Auth
from langchain.llms.bedrock import Bedrock
from botocore.awsrequest import AWSRequest
import streamlit as st
import re
import numpy as np
from sklearn.metrics import ndcg_score,dcg_score
from sklearn import preprocessing as pre
import invoke_models

bedrock_ = boto3.client('bedrock-runtime',region_name='us-east-1')

inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}
textgen_llm = Bedrock(
    model_id="anthropic.claude-v2:1",
    client=bedrock_,
    model_kwargs=inference_modifier,
)


#@st.cache_data
def eval(question, answers):
    #if()
    search_results: str = ""
    prompt: str = """Human: You are a grader assessing relevance of a retrieved document to a user question. \n 
    The User question and Retrieved documents are provided below. The Retrieved documents are retail product descriptions that the human is looking for. \n
    It does not need to be a stringent test. The goal is to filter out totally irrelevant product retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    
    <User question>
    {}
    </User question>

    <Retrieved document>
    {}
    </Retrieved document>

    Now based on the information provided above, for every given Retrieved document, provide the index of the document, it's score out of 5 based on relevance with the User question, is it relevant or not as true or false, reason why you this is relevant or not, in json format,
    
    Answer:
    """
    #Finally, as the last line of your response, write the relevant indexes as a comma separated list in a line.


    query = question[0]['question']
    index_ = 0
    for i in answers[0]['answer']:
        desc = i['caption']+ "."+ i['desc']
        search_results += f"Index: {index_}, Description: {desc}\n\n"
        index_ = index_+1
    prompt = prompt.format(query, search_results)
    # print(answers[0]['answer'])
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(prompt)

    response = textgen_llm(prompt)
    #invoke_models.invoke_llm_model(prompt,False)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(response)
    inter_trim =response.split("[")[1]
    final_out = json.loads('{"results":['+inter_trim.split("]")[0]+']}')
    #final_out_sorted_desc  = sorted(final_out['results'], key=lambda d: d['Score'],reverse=True) 
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(final_out_sorted_desc)
    
    #true_relevance = np.asarray([[10, 0, 0, 1, 5]])
    llm_scores = []
    current_scores = []
    for idx,i in enumerate(answers[0]['answer']):
        if('relevant' in final_out['results'][idx]):
            relevance = final_out['results'][idx]['relevant']
        else:
            relevance = final_out['results'][idx]['Relevant']
        if('score' in final_out['results'][idx]):
            score_ = final_out['results'][idx]['score']
        else:
            score_ = final_out['results'][idx]['Score']
        i['relevant'] = relevance  
        llm_scores.append(score_)
        current_scores.append(i['score'])
        
        
    
    #   llm_scores.sort(reverse = True)
    x = np.array(llm_scores)
    x = x.reshape(-1, 1)
    x_norm = (pre.MinMaxScaler().fit_transform(x)).flatten().tolist()
    
    y = np.array(current_scores)
    y = y.reshape(-1, 1)
    y_norm = (pre.MinMaxScaler().fit_transform(y)).flatten().tolist()

   
    st.session_state.answers = answers
    
    # print(x_norm)
    # print(y_norm)
    
    dcg = dcg_score(np.asarray([llm_scores]),np.asarray([current_scores]))
    # print("DCG score : ", dcg)
  
    # IDCG score
    idcg = dcg_score(np.asarray([llm_scores]),np.asarray([llm_scores]))
    # print("IDCG score : ", idcg)
    
    # Normalized DCG score
    ndcg = dcg
    
    # print(st.session_state.input_ndcg)
    if(ndcg > st.session_state.input_ndcg and st.session_state.input_ndcg!=0.0):
        st.session_state.ndcg_increase = "&uarr;~"+str('%.3f'%(ndcg-st.session_state.input_ndcg ))
    elif(ndcg < st.session_state.input_ndcg):
        st.session_state.ndcg_increase = "&darr;~"+str('%.3f'%(st.session_state.input_ndcg - ndcg))
    else:
        st.session_state.ndcg_increase = " ~ "
        
            
    
    st.session_state.input_ndcg = ndcg#round(ndcg_score(np.asarray([x_norm]), np.asarray([y_norm]), k=st.session_state.input_K),2)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(st.session_state.answers)
        