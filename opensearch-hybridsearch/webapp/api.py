import json
from urllib.parse import urlparse, urlencode, parse_qs

import re

import requests
import boto3
from boto3 import Session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import botocore.session


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
    print("=========")
    print(body)
    search_type = (json.loads(prompt))['searchType']
    print("=========")
    method = "post"
    url = "https://bwwj5hqbym7w245urmrijqrifa0gzmoc.lambda-url.us-east-1.on.aws/" #https://bwwj5hqbym7w245urmrijqrifa0gzmoc.lambda-url.us-east-1.on.aws/
    #https://$query_invoke_URL_cmd.execute-api.us-east-1.amazonaws.com/prod/lambda
    r = requests.post(url, headers= signing_headers(method,url,body), data=body)

    print("*********")
    print(r.text)

    #{"Content-Type": "application/json; charset=utf-8"}
    #signing_headers(method,url,body)
    response_ = json.loads(json.loads(r.text))
    print(response_.keys())
    docs = response_['hits']['hits']
    print(docs)
    arr = []
    if(search_type == 'Multi-modal Search'):
        key_ = 'image_description'
    else:
        key_ = 'caption'
    #filter_out = 0
    for doc in docs:
        # if('b5/b5319e00' in doc['_source']['image_s3_url'] ):
        #     filter_out +=1
        #     continue
        arr.append({"desc":doc['_source'][key_],"image_url":doc['_source']['image_s3_url']})

    return arr
