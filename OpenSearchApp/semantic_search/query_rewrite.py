import json
import os
import sys
import boto3
import amazon_rekognition
from botocore.config import Config
import getpass
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import streamlit as st
from langchain.schema import Document
from langchain_community.vectorstores import OpenSearchVectorSearch,ElasticsearchStore
from requests_aws4auth import AWS4Auth
from requests.auth import HTTPBasicAuth
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

from langchain.retrievers.self_query.opensearch import OpenSearchTranslator

from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
bedrock_params = {
    "max_tokens_to_sample":2048,
    "temperature":0.0001,
    "top_k":250,
    "top_p":1,
    "stop_sequences":["\\n\\nHuman:"]
}
bedrock_region="us-east-1"

#boto3_bedrock = boto3.client(service_name="bedrock-runtime", endpoint_url=f"https://bedrock-runtime.{bedrock_region}.amazonaws.com")
boto3_bedrock = boto3.client(service_name="bedrock-runtime", config=Config(region_name=bedrock_region))

bedrock_titan_llm = Bedrock(model_id="anthropic.claude-instant-v1", client=boto3_bedrock)
bedrock_titan_llm.model_kwargs = bedrock_params

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.embeddings import BedrockEmbeddings

bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',client=boto3_bedrock)

from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
aos_host = 'search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com'
credentials = boto3.Session().get_credentials()
auth = AWS4Auth(credentials.access_key, credentials.secret_key, 'us-east-1', 'es', session_token=credentials.token)

aos_client = OpenSearch(
    hosts = [{'host': aos_host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)
os_domain_ep = aos_host


metadata_field_info_ = [
    AttributeInfo(
        name="price",
        description="Cost of the product",
        type="string",
    ),
    AttributeInfo(
        name="style",
        description="The style of the product",
        type="string",
    ),
    AttributeInfo(
        name="category",
        description="The category of the product, the available categories are apparel, footwear, outdoors, electronics, beauty, jewelry, accessories, housewares, homedecor, furniture, seasonal, floral, books, groceries, instruments, tools, hot dispensed, cold dispensed, food service and salty snacks",
        type="string",
    ),
    AttributeInfo(
        name="current_stock",
        description="The available quantity of the product",
        type="string",
    ),
    AttributeInfo(
        name="gender_affinity", 
        description="The gender that the product relates to, the choices are Male and Female", 
        type="string"
    ),
    AttributeInfo(
        name="caption", 
        description="The short description of the product", 
        type="string"
    ),
    AttributeInfo(
        name="description", 
        description="The detailed description of the product", 
        type="string"
    ),
    AttributeInfo(
        name="color", 
        description="The color of the product", 
        type="string"
    )
]
document_content_description_ = "Brief summary of a retail product"

open_search_vector_store = OpenSearchVectorSearch(
                                    index_name="retail-ml-search-index",#"self-query-rewrite-retail",
                                    embedding_function=bedrock_embeddings,
                                    opensearch_url=os_domain_ep,
                                    http_auth=auth
                                    )  

# retriever = SelfQueryRetriever.from_llm(
#     bedrock_titan_llm, open_search_vector_store, document_content_description_, metadata_field_info_, verbose=True
# )

# res = retriever.get_relevant_documents("bagpack for men")

# st.write(res)


prompt = get_query_constructor_prompt(
    document_content_description_,
    metadata_field_info_,
)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | bedrock_titan_llm | output_parser

def get_new_query_res(query):
    if(query == ""):
        query = st.session_state.input_rekog_label
    if(st.session_state.input_is_rewrite_query == 'enabled'):

        query_struct = query_constructor.invoke(
            {
                "query": query
            }
        )
        print("***prompt****")
        print(prompt)
        print("******query_struct******")
        print(query_struct)

        opts = OpenSearchTranslator()
        query_ = json.loads(json.dumps(opts.visit_structured_query(query_struct)[1]['filter']).replace("must","should"))#.replace("must","should")
        if('bool' in query_ and 'should' in query_['bool']):
            query_['bool']['should'].append({
                    "match": {
                    
                        "rekog_description_plus_original_description": query
                    
                    }
                })
        else:
            query_['bool']['should'] = {
                    "match": {
                    
                        "rekog_description_plus_original_description": query
                    
                    }
                }
        
        # def find_by_key(data, target):
        #     for key, value in data.items():
        #         if isinstance(value, dict):
        #             yield from find_by_key(value, target)
        #         elif key == target:
        #             yield value
        # for x in find_by_key(query_, "metadata.category.keyword"):
        #     imp_item = x
        
        imp_item = ""
        if("bool" in query_ and  'should' in query_['bool']):
            for i in query_['bool']['should']:
                if("term" in i.keys()):
                    if("metadata.category.keyword" in i["term"]):
                        imp_item = imp_item + i["term"]["metadata.category.keyword"]+ " "
                    if("metadata.style.keyword" in i["term"]):
                        imp_item = imp_item + i["term"]["metadata.style.keyword"]+ " "
                if("match" in i.keys()):
                    if("metadata.category.keyword" in i["match"]):
                        imp_item = imp_item + i["match"]["metadata.category.keyword"]+ " "
                    if("metadata.style.keyword" in i["match"]):
                        imp_item = imp_item + i["match"]["metadata.style.keyword"]+ " "
        else:
            if("term" in query_):
                    if("metadata.category.keyword" in query_):
                        imp_item = imp_item + query_["metadata.category.keyword"] + " "
                    if("metadata.style.keyword" in query_):
                        imp_item = imp_item + query_["metadata.style.keyword"]+ " "
            if("match" in query_):
                    if("metadata.category.keyword" in query_):
                        imp_item = imp_item + query_["metadata.category.keyword"]+ " "
                    if("metadata.style.keyword" in query_):
                        imp_item = imp_item + query_["metadata.style.keyword"]+ " "
            
                    
        if(imp_item == ""):
            imp_item = query
            
        ps = PorterStemmer()        
        def stem_(sentence):
            words = word_tokenize(sentence)
            
            words_stem = ""

            for w in words:
                words_stem = words_stem +" "+ps.stem(w)
            return words_stem.strip()
        
        imp_item = stem_(imp_item)
        print("imp_item---------------")
        print(imp_item)
        
        query_['bool']['must']={
                    "multi_match": {
                    
                        "query": imp_item.strip(),
                      "fields":['description','rekog_all^3', "category","style"]
                    
                    }
                }
       
            
        #query_['bool']["minimum_should_match"] = 1
            
        st.session_state.input_rewritten_query = {"query":query_}
    if(st.session_state.input_rekog_label!=""):
        amazon_rekognition.call(st.session_state.input_text,st.session_state.input_rekog_label)
    #return searchWithNewQuery(st.session_state.input_rewritten_query)

# def searchWithNewQuery(new_query):
#     response = aos_client.search(
#         body = new_query,
#         index = "demo-retail-rekognition"#'self-query-rewrite-retail',
#         #pipeline = 'RAG-Search-Pipeline'
#     )
    
#     hits = response['hits']['hits']
#     print("rewrite-------------------------")
#     arr = []
#     for doc in hits:
#         # if('b5/b5319e00' in doc['_source']['image_s3_url'] ):
#         #     filter_out +=1
#         #     continue
        
#         res_ = {"desc":doc['_source']['text'],"image_url":doc['_source']['metadata']['image_s3_url']}
#         if('highlight' in doc):
#             res_['highlight'] = doc['highlight']['text']
#         # if('caption_embedding' in doc['_source']):
#         #     res_['sparse'] = doc['_source']['caption_embedding']
#         # if('query_sparse' in response_ and len(arr) ==0 ):
#         #     res_['query_sparse'] = response_["query_sparse"]
#         res_['id'] = doc['_id']
#         res_['score'] = doc['_score']
#         res_['title'] = doc['_source']['text']
           
#         arr.append(res_)
            

    
#     return arr
    
    




