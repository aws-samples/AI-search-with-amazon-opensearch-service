import json
import os
import sys
import boto3
import amazon_rekognition
from botocore.config import Config
import getpass
import nltk
nltk.download('punkt')
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
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
import utilities.invoke_models as invoke_models



bedrock_params = {
    "max_tokens_to_sample":2048,
    "temperature":0.0001,
    "top_k":250,
    "top_p":1,
    "stop_sequences":["\\n\\nHuman:"]
}
bedrock_region=st.session_state.REGION
boto3_bedrock = boto3.client(service_name="bedrock-runtime", config=Config(region_name=bedrock_region))

bedrock_titan_llm = Bedrock(model_id="anthropic.claude-instant-v1", client=boto3_bedrock)
bedrock_titan_llm.model_kwargs = bedrock_params
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',client=boto3_bedrock)


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

schema = """{{
    "content": "Brief summary of a retail product",
    "attributes": {{
    "category": {{
        "description": "The category of the product, the available categories are apparel, footwear, outdoors, electronics, beauty, jewelry, accessories, housewares, homedecor, furniture, seasonal, floral, books, groceries, instruments, tools, hot dispensed, cold dispensed, food service and salty snacks",
        "type": "string"
    }},
    "gender_affinity": {{
        "description": "The gender that the product relates to, the choices are Male and Female", 
        "type": "string"
    }},
    "price": {{
        "description": "Cost of the product",
        "type": "double"
    }},
    "description": {{
        "description": "The detailed description of the product",
        "type": "string"
    }},
    "color": {{
        "description": "The color of the product",
        "type": "string"
    }},
     "caption": {{
        "description": "The short description of the product",
        "type": "string"
    }},
     "current_stock": {{
        "description": "The available quantity of the product in stock for sale",
        "type": "integer"
    }},
     "style": {{
        "description": "The style of the product",
        "type": "string"
    }}
}}
}}"""
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
        name="product_description", 
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
                                    index_name="retail-ml-search-index",
                                    embedding_function=bedrock_embeddings,
                                    opensearch_url=os_domain_ep,
                                    http_auth=auth
                                    )  

examples = [
    {  "i":1,
        "data_source": schema,
        "user_query": "black shoes for men",
        "structured_request": """{{
    "query": "shoes",
    "filter": "and(eq(\"color\", \"black\"),  eq(\"category\", \"footwear\")), eq(\"gender_affinity\", \"Male\")"
            }}""",
    },
    
        {  "i":2,
        "data_source": schema,
        "user_query": "black or brown jackets for men under 50 dollars",
        "structured_request": """{{
    "query": "jackets",
    "filter": "and(eq(\"style\", \"jacket\"), or(eq(\"color\", \"brown\"),eq(\"color\", \"black\")),eq(\"category\", \"apparel\"),eq(\"gender_affinity\", \"male\"),lt(\"price\", \"50\"))"
            }}""",
    },
          {  "i":2,
        "data_source": schema,
        "user_query": "trendy handbags for women",
        "structured_request": """{{
    "query": "handbag",
    "filter": "and(eq(\"style\", \"bag\") ,eq(\"category\", \"accessories\"),eq(\"gender_affinity\", \"female\"))"
            }}""",
    }
]


example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)
example_prompt=PromptTemplate(input_variables=['data_source', 'i', 'structured_request', 'user_query'],
template='<< Example {i}. >>\nData Source:\n{data_source}\n\nUser Query:\n{user_query}\n\nStructured Request:\n{structured_request}\n')

prefix_ = """
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
"""

suffix_ = """<< Example 3. >>
Data Source:
{schema}

User Query:
{query}

Structured Request:
"""

prompt_ = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix=suffix_,
    prefix=prefix_,
    input_variables=["query","schema"],
)




############ Langchain's self qeury retriever method ##########
# retriever = SelfQueryRetriever.from_llm(
#     bedrock_titan_llm, open_search_vector_store, document_content_description_, metadata_field_info_, verbose=True
# )

# res = retriever.get_relevant_documents("bagpack for men")

# st.write(res)

######### use this for self query retriever ########
# prompt = get_query_constructor_prompt(
#     document_content_description_,
#     metadata_field_info_,
# )
# output_parser = StructuredQueryOutputParser.from_components()
# query_constructor = prompt | bedrock_titan_llm | output_parser
############ Langchain's self qeury retriever method ##########


def get_new_query_res(query):
    field_map = {'Price':'price','Gender':'gender_affinity','Category':'category','Style':'style','Color':'color'}
    field_map_filter = {key: field_map[key] for key in st.session_state.input_must}
    if(query == ""):
        query = st.session_state.input_rekog_label
    if(st.session_state.input_is_rewrite_query == 'enabled'):

        # query_struct = query_constructor.invoke(
        #     {
        #         "query": query
        #     }
        # )
        # print("***prompt****")
        # print(prompt)
        # print("******query_struct******")
        # print(query_struct)

        res = invoke_models.invoke_llm_model( prompt_.format(query=query,schema = schema)  ,False)
        inter_query = res[7:-3].replace('\\"',"'").replace("\n","")
        print("inter_query")
        print(inter_query)
        query_struct = StructuredQueryOutputParser.from_components().parse(inter_query)  
        print("query_struct")
        print(query_struct)
        opts = OpenSearchTranslator()
        result_query_llm = opts.visit_structured_query(query_struct)[1]['filter']
        print(result_query_llm)
        draft_new_query = {'bool':{'should':[],'must':[]}}
        if('bool' in result_query_llm and ('must' in result_query_llm['bool'] or 'should' in result_query_llm['bool'])):
            #draft_new_query['bool']['should'] = []
            if('must' in result_query_llm['bool']):
                for q in result_query_llm['bool']['must']:
                    old_clause = list(q.keys())[0]
                    if(old_clause == 'term'):
                        new_clause = 'match'
                    else:
                        new_clause = old_clause
                    q_dash = {}
                    q_dash[new_clause] = {}
                    long_field = list(q[old_clause].keys())[0]
                    #print(long_field)
                    get_attr = long_field.split(".")[1]
                    #print(get_attr)
                    q_dash[new_clause][get_attr] = q[old_clause][long_field]
                    #print(q_dash)
                    if(get_attr in list(field_map_filter.values())):
                        draft_new_query['bool']['must'].append(q_dash)
                    else:
                        draft_new_query['bool']['should'].append(q_dash)
            # if('should' in result_query_llm['bool']):
            #     for q_ in result_query_llm['bool']['must']:
            #         q__dash = json.loads(json.dumps(q_).replace('term','match'  ))
            #         clause = list(q__dash.keys())[0]category
            #         long_field = list(q__dash[clause].keys())[0]
            #         get_attr = long_field.split(".")[1]
            #         q__dash[clause][get_attr] = q__dash[clause][long_field]
            #         draft_new_query['bool']['should'].append(q__dash)
                
        #print(draft_new_query)    
        query_ = draft_new_query#json.loads(json.dumps(opts.visit_structured_query(query_struct)[1]['filter']).replace("must","should"))#.replace("must","should")
        
        # if('bool' in query_ and 'should' in query_['bool']):
        #     query_['bool']['should'].append({
        #             "match": {
                    
        #                 "rekog_description_plus_original_description": query
                    
        #             }
        #         })
        # else:
        #     query_['bool']['should'] = {
        #             "match": {
                    
        #                 "rekog_description_plus_original_description": query
                    
        #             }
        #         }
        
        # def find_by_key(data, target):
        #     for key, value in data.items():
        #         if isinstance(value, dict):
        #             yield from find_by_key(value, target)
        #         elif key == target:
        #             yield value
        # for x in find_by_key(query_, "metadata.category.keyword"):
        #     imp_item = x
        
        
        ###### find the main subject of the query
        #imp_item = ""
        # if("bool" in query_ and  'should' in query_['bool']):
        #     for i in query_['bool']['should']:
        #         if("term" in i.keys()):
        #             if("metadata.category.keyword" in i["term"]):
        #                 imp_item = imp_item + i["term"]["metadata.category.keyword"]+ " "
        #             if("metadata.style.keyword" in i["term"]):
        #                 imp_item = imp_item + i["term"]["metadata.style.keyword"]+ " "
        #         if("match" in i.keys()):
        #             if("metadata.category.keyword" in i["match"]):
        #                 imp_item = imp_item + i["match"]["metadata.category.keyword"]+ " "
        #             if("metadata.style.keyword" in i["match"]):
        #                 imp_item = imp_item + i["match"]["metadata.style.keyword"]+ " "
        # else:
        #     if("term" in query_):
        #             if("metadata.category.keyword" in query_):
        #                 imp_item = imp_item + query_["metadata.category.keyword"] + " "
        #             if("metadata.style.keyword" in query_):
        #                 imp_item = imp_item + query_["metadata.style.keyword"]+ " "
        #     if("match" in query_):
        #             if("metadata.category.keyword" in query_):
        #                 imp_item = imp_item + query_["metadata.category.keyword"]+ " "
        #             if("metadata.style.keyword" in query_):
        #                 imp_item = imp_item + query_["metadata.style.keyword"]+ " "
        ###### find the main subject of the query
        imp_item = (opts.visit_structured_query(query_struct)[0]).replace(",","")
            
                    
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
        if('must' in query_['bool']):
            query_['bool']['must'].append({
                    "simple_query_string": {
                    
                        "query": imp_item.strip(),
                      "fields":['product_description',"style","caption"]#'rekog_all^3'
                    
                    }
                    #"match":{"description":imp_item.strip()}
                })
        else:
            query_['bool']['must']={
                    "multi_match": {
                    
                        "query": imp_item.strip(),
                      "fields":['product_description',"style"]#'rekog_all^3'
                    
                    }
                    #"match":{"description":imp_item.strip()}
                }
       
            
        #query_['bool']["minimum_should_match"] = 1
            
        st.session_state.input_rewritten_query = {"query":query_}
        print(st.session_state.input_rewritten_query)
    # if(st.session_state.input_rekog_label!="" and query!=st.session_state.input_rekog_label):
    #     amazon_rekognition.call(st.session_state.input_text,st.session_state.input_rekog_label)
    
    
    # #return searchWithNewQuery(st.session_state.input_rewritten_query)

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
    
    




