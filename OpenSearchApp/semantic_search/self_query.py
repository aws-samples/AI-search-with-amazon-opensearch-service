import json
import os
import sys
import boto3
from botocore.config import Config
import getpass
import os
import streamlit as st

from langchain.schema import Document
from langchain_community.vectorstores import OpenSearchVectorSearch,ElasticsearchStore
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
aos_host = 'https://search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com'

auth = ('prasadnu', '@Palamalai1') #### input credentials
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
        description="The retail category of the product",
        type="string",
    ),
    AttributeInfo(
        name="current_stock",
        description="The available quantity of the product",
        type="string",
    ),
    AttributeInfo(
        name="gender_affinity", 
        description="The gender that the product can be used by", 
        type="string"
    ),
    AttributeInfo(
        name="name", 
        description="The short description of the product", 
        type="string"
    )
]
document_content_description_ = "Brief summary of a retail product"



open_search_vector_store = OpenSearchVectorSearch(
                                    index_name="self-query-retail-demo-2",
                                    embedding_function=bedrock_embeddings,
                                    opensearch_url=os_domain_ep,
                                    http_auth=auth
                                    )  

retriever = SelfQueryRetriever.from_llm(
    bedrock_titan_llm, open_search_vector_store, document_content_description_, metadata_field_info_, verbose=True
)


# This example only specifies a relevant query
res = retriever.get_relevant_documents("bagpack for men")


st.write(res)
