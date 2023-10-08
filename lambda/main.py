import json
import os
from typing import Dict
import requests
import chain
import config


def handler(event, context): 
    
    body = json.loads(event["body"])
    
    # validate_response = validate_inputs(body)
    # if validate_response:
    #     return validate_response

   
    
    # prompt = "what is the recommended shard size?"
    # session_id = "7859484"

    prompt = body['prompt']
    session_id = (body["session_id"]).strip()

    print(f"prompt is {prompt}")
    print(f"session_id is {session_id}")
    
    response, session_id = chain.run(
        api_key=get_api_key(), 
        session_id=session_id, 
        prompt=prompt
    )

    print("response inside main")
    print( {
        "response": response,
        "session_id": session_id
    })
    return build_response({
        "response": response,
        "session_id": session_id
    })


def validate_inputs(body: Dict):
    for input_name in ['prompt', 'session_id']:
        if input_name not in body:
            return build_response({
                "status": "error",
                "message": f"{input_name} missing in payload"
            })
    return ""

def build_response(body: Dict):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(body)
    }


def get_api_key():
    """Fetches the api keys saved in Secrets Manager"""

    # headers = {"X-Aws-Parameters-Secrets-Token": os.environ.get('AWS_SESSION_TOKEN')}
    # secrets_extension_endpoint = "http://localhost:2773" + \
    # "/secretsmanager/get?secretId=" + \
    # config.config.API_KEYS_SECRET_NAME
  
    # r = requests.get(secrets_extension_endpoint, headers=headers)
    # secret = json.loads(json.loads(r.text)["SecretString"])

    return "openAI key"
