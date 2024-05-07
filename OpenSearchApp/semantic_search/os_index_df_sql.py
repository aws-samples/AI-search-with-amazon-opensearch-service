# imports to demonstrate DataFrame support
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import opensearch_py_ml as oml
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import invoke_models
import json
import streamlit as st


# Import standard test settings for consistent results
from opensearch_py_ml.conftest import *
import boto3
from requests_aws4auth import AWS4Auth
from requests.auth import HTTPBasicAuth
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_models import BedrockChat
import boto3
region = 'us-east-1'
bedrock_runtime_client = boto3.client('bedrock-runtime',region_name=region)






################ OpenSearch Py client #####################
    
credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, "us-east-1", "es")

ospy_client = OpenSearch(
    hosts = [{'host': 'search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com', 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)



def sql_process(question):
    oml_retail = oml.DataFrame(ospy_client, 'retail-ml-search-index')

    pd_retail = oml.opensearch_to_pandas(oml_retail)
    
    df = pd_retail.drop(columns=['desc_embedding_bedrock-multimodal', 'desc_embedding_bedrock-text','desc_embedding_sparse'])
    
    # engine = create_engine("sqlite:///retail_db.db")
    # df.to_sql("retail-ml-search-index", engine, index=False)
    # db = SQLDatabase(engine=engine)
    
    db = SQLDatabase.from_uri("sqlite:///retail_db.db")
    print(db.dialect)
    print(db.get_usable_table_names())
    

    bedrock_params = {
    "max_tokens":2048,
    "temperature":0.0001,
    "top_k":150,
    "top_p":0.7,
    "stop_sequences":["\\n\\nHuman:"]
    }
    
    model = BedrockChat(
    client=bedrock_runtime_client,
    model_id='anthropic.claude-3-sonnet-20240229-v1:0',
    model_kwargs=bedrock_params,
    streaming=False
    )
    agent_executor = create_sql_agent(model, db=db, verbose=True,handle_parsing_errors=True)
    question_prompt = "Display only the final sql query that gives top 10 results for the question: "+question+". " 
    try:
        res = agent_executor.invoke({"input": question_prompt})
    except ValueError as e:
        print( e)
        final_sql = str(e).split(":")[-1].replace('\n',' ').replace('`','')
    else:
        final_sql = json.loads(res)['output'].split(";")[0]
    st.session_state.input_sql_query = final_sql
    #for i in res['output']:
        