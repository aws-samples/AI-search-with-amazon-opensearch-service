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
#from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth
from datetime import datetime
import boto3
import subprocess

client = boto3.client('s3') # example client, could be any

proc=subprocess.Popen('ec2-metadata -z | grep -Po "(us|sa|eu|ap)-(north|south|central)?(east|west)?-[0-9]+"', shell=True, stdout=subprocess.PIPE, )
region=(proc.communicate()[0].decode("utf-8")).strip()
LAMBDA_INTERVAL=60


client = boto3.client('resourcegroupstaggingapi',region_name=region)

response = client.get_resources(
    TagFilters=[
        {
            'Key': 'app',
            'Values': [
                'hybrid-search',
            ]
        }
    ]
)


print(response)
opensearch_arn = response['ResourceTagMappingList'][0]['ResourceARN']
opensearch_ = boto3.client('opensearch',region_name=region)
response = opensearch_.describe_domain(
    DomainName=opensearch_arn.split("/")[1]
)
DOMAIN_ENDPOINT = response['DomainStatus']['Endpoint']


REGION = region #'us-west-2'#


#MODEL_ID = os.environ['MODEL_ID'] #'P_vh7YsBNIVobUP3w7RJ' #

current_date_time = (datetime.now()).isoformat()
today_ = datetime.today().strftime('%Y-%m-%d')

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, 'es', session_token=credentials.token)
path =  "_plugins/_ml/models/_search"
host = 'https://'+DOMAIN_ENDPOINT+'/'
url = host + path
headers = {"Content-Type": "application/json"}
payload = {
  "query": {
    "match": {"name":"embedding-gpt"}
  },
  
  "sort": [{
            "last_deployed_time": {
                "order": "desc"
            }
        }]
}

r = requests.post(url, auth=awsauth, json=payload, headers=headers)
print(r.status_code)

print(r.text)
print(type(r.text))

MODEL_ID = json.loads(r.text)["hits"]["hits"][0]["_id"]

# Lambda handler
def handler(event):
    print("^^^^^6")
    print(event)

    #{'inputs': '{"CombineType": "arithmetic_mean", "weight": 0.5, "text": "wine glass", "NormType": "min_max", "searchType": "Keyword Search"}', 'session_id': ''}input_ = 
    print("------")
    print(event)
    print("------")
    #print(event['body'])

    input_ = json.loads(event["inputs"])
    print("*********")
    print(input_)

    norm_type = input_["NormType"]
    combine_type = input_["CombineType"]
    semantic_weight = input_["weight"]
    search_type = input_["searchType"]
    query = input_["text"]
    k_ = input_["K"]

    if(search_type == 'Keyword Search'):
        semantic_weight = 0.0
    if(search_type == 'Vector Search'):
        semantic_weight = 1.0

        


    path = "_search/pipeline/nlp-search-pipeline" 
    host = 'https://'+DOMAIN_ENDPOINT+'/'
    url = host + path

    payload = {
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
                "weights": [1-semantic_weight,semantic_weight]
                }
            }
            }
        }
        ]
    }

    headers = {"Content-Type": "application/json"}


    r = requests.put(url, auth=awsauth, json=payload, headers=headers)
    print(r.status_code)
    print(r.text)


    path = "nlp-image-search-index/_search?search_pipeline=nlp-search-pipeline" 
    url = host + path

    payload = {
        "_source": {
        "exclude": [
            "caption_embedding"
        ]
        },
        "query": {
        "hybrid": {
            "queries": [
            {
                "match": {
                "caption": {
                    "query": query
                }
                }
            },
            {
                "neural": {
                "caption_embedding": {
                    "query_text": query,
                    "model_id": MODEL_ID,
                    "k": k_
                }
                }
            }
            ]
        }
        },"size":k_
    }

    r = requests.get(url, auth=awsauth, json=payload, headers=headers)
    print(r.status_code)
    print(r.text)
    return r.text

