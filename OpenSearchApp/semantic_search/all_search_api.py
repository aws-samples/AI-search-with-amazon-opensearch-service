import json
from urllib.parse import urlparse, urlencode, parse_qs

import re

import requests
import boto3
from boto3 import Session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import botocore.session
import all_search



def signing_headers(method, url_string, body):
    region = url_string.split(".")[2]
    url = urlparse(url_string)
    path = url.path or '/'
    querystring = ''
    if url.query:
        querystring = '?' + urlencode(
            parse_qs(url.query, keep_blank_values=True), doseq=True)

    safe_url = url.scheme + '://' + url.netloc.split(
        ':')[0] + path + querystring
    request = AWSRequest(method=method.upper(), url=safe_url, data=body)

    session = botocore.session.Session()
    sigv4 = SigV4Auth(session.get_credentials(), "lambda", region)
    sigv4.add_auth(request)

    header_ = dict(request.headers.items())
    
    header_["Content-Type"] = "application/json; charset=utf-8"
    
    print(header_)

    return dict(header_)

def call(prompt: str, session_id: str):
    body = json.dumps({
        "inputs": prompt,
        "session_id": session_id
    })

    search_type = (json.loads(prompt))['searchType']
    
    method = "post"
    url = "https://bwwj5hqbym7w245urmrijqrifa0gzmoc.lambda-url.us-east-1.on.aws/"

    r = requests.post(url, headers= signing_headers(method,url,body), data=body)

    response_ = json.loads(json.loads(r.text))
    print(response_.keys())
    docs = response_['hits']['hits']
    #print(docs)
    arr = []
    dup = []
    if('Multi-modal Search' in search_type ):
        key_ = 'image_description'
    else:
        key_ = 'caption'
    #filter_out = 0
    
    for doc in docs:
        # if('b5/b5319e00' in doc['_source']['image_s3_url'] ):
        #     filter_out +=1
        #     continue
        
        if(doc['_source']['image_s3_url'] not in dup):
            res_ = {"desc":doc['_source'][key_],"image_url":doc['_source']['image_s3_url']}
            if('highlight' in doc):
                res_['highlight'] = doc['highlight'][key_]
            if('caption_embedding' in doc['_source']):
                #print("@@@@@@@@@@@@@@@@@@@@@")
                res_['sparse'] = doc['_source']['caption_embedding']
            if('query_sparse' in response_ and len(arr) ==0 ):
                res_['query_sparse'] = response_["query_sparse"]
            res_['id'] = doc['_id']
            res_['score'] = doc['_score']
            res_['title'] = doc['_source']['caption']
           
            arr.append(res_)
            dup.append(doc['_source']['image_s3_url'])

    size_ = json.loads(prompt)['K']

    # print("size:::::"+str(size_))
    # print(arr)

    return arr[0:size_]