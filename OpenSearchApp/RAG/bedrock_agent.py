import boto3
import json
import time
import zipfile
from io import BytesIO
import uuid
import pprint
import logging
print(boto3.__version__)
from PIL import Image 
import os
import base64
import re
import requests 
import utilities.re_ranker as re_ranker
import utilities.invoke_models as invoke_models
import streamlit as st
import time as t
import botocore.exceptions

if "inputs_" not in st.session_state:
    st.session_state.inputs_ = {}

parent_dirname = "/".join((os.path.dirname(__file__)).split("/")[0:-1])
session = boto3.session.Session()
region = st.session_state.REGION 
print(region)
sts_client = boto3.client('sts')
account_id = sts_client.get_caller_identity()["Account"]
# setting logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
# getting boto3 clients for required AWS services

iam_client = boto3.client('iam')
lambda_client = boto3.client('lambda',region_name=region)
bedrock_agent_client = boto3.client('bedrock-agent',region_name=region)
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime',region_name=region)
# configuration variables
suffix = f"{region}-{account_id}"
agent_name = "shopping-assistant-function-def"
agent_bedrock_allow_policy_name = f"{agent_name}-ba-{suffix}"
agent_role_name = f'AmazonBedrockExecutionRoleForAgents_shopping'
agent_foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"
agent_description = "Agent for providing HR assistance to manage vacation time"
agent_instruction = "You are a shopping assistant, helping shoppers to pick right shoes based on their needs referring to the product catalog"
agent_action_group_name = "ShoppingActionGroup"
agent_action_group_description = "Actions for getting the query from the user, retrieve relevant items from the product catalog and provide recommendations to the user"
agent_alias_name = f"{agent_name}-alias"
lambda_function_role = f'{agent_name}-lambda-role'
lambda_function_name = f'{agent_name}-{suffix}'


enable_trace:bool = True
end_session:bool = False

agentAliasId = st.session_state.BedrockAgentAlias.split("|")[1]   
agentId = st.session_state.BedrockAgentAlias.split("|")[0]  

def delete_memory():
    response = bedrock_agent_runtime_client.delete_agent_memory(
    agentAliasId=agentAliasId,
    agentId=agentId
    )
    
def query_(inputs):
    ## create a random id for session initiator id
    
    
    # invoke the agent API
    agentResponse = bedrock_agent_runtime_client.invoke_agent(
        inputText=inputs['shopping_query'],
        agentId=agentId,
        agentAliasId=agentAliasId, 
        sessionId=st.session_state.session_id_,
        enableTrace=enable_trace, 
        endSession= end_session
    )

    logger.info(pprint.pprint(agentResponse))
    print("***agent*****response*********")
    print(agentResponse)
    event_stream = agentResponse['completion']
    total_context = []
    last_tool = ""
    last_tool_name = ""
    agent_answer = ""
    try:
        for event in event_stream:
            print("***event*********")
            print(event)
            # if 'chunk' in event:
            #     data = event['chunk']['bytes']
            #     print("***chunk*********")
            #     print(data)
            #     logger.info(f"Final answer ->\n{data.decode('utf8')}")
            #     agent_answer_ = data.decode('utf8')
            #     print(agent_answer_)
            if 'trace' in event: 
                print("trace*****total*********")
                print(event['trace'])
                if('orchestrationTrace' not in event['trace']['trace']):
                    continue
                orchestration_trace = event['trace']['trace']['orchestrationTrace']
                total_context_item = {}
                if('modelInvocationOutput' in orchestration_trace and '<tool_name>' in orchestration_trace['modelInvocationOutput']['rawResponse']['content']):
                    total_context_item['tool'] = orchestration_trace['modelInvocationOutput']['rawResponse']
                if('rationale' in orchestration_trace):
                    total_context_item['rationale'] = orchestration_trace['rationale']['text']
                if('invocationInput' in orchestration_trace):
                    total_context_item['invocationInput'] = orchestration_trace['invocationInput']['actionGroupInvocationInput']
                    last_tool_name = total_context_item['invocationInput']['function']
                if('observation'  in orchestration_trace):
                    print("trace****observation******")
                    total_context_item['observation'] = event['trace']['trace']['orchestrationTrace']['observation']
                    tool_output_last_obs = event['trace']['trace']['orchestrationTrace']['observation']
                    print(tool_output_last_obs)
                    if(tool_output_last_obs['type'] == 'ACTION_GROUP'):
                        last_tool = tool_output_last_obs['actionGroupInvocationOutput']['text']
                    if(tool_output_last_obs['type'] == 'FINISH'):   
                        agent_answer = tool_output_last_obs['finalResponse']['text'].replace(account_id + "-ml-search","xxxxx-ml-search")
                if('modelInvocationOutput' in orchestration_trace and '<thinking>' in orchestration_trace['modelInvocationOutput']['rawResponse']['content']):
                    total_context_item['thinking'] = orchestration_trace['modelInvocationOutput']['rawResponse']
                if(total_context_item!={}):
                    removed_s3_bucket = (json.dumps(total_context_item)).replace(account_id + "-ml-search","xxxxx-ml-search")
                    total_context.append(json.loads(removed_s3_bucket))
        print("total_context------")
        print(total_context)    
    except botocore.exceptions.EventStreamError as error:
        raise error
        # t.sleep(2)
        # query_(st.session_state.inputs_)     
                
            # if 'chunk' in event:
            #     data = event['chunk']['bytes']
            #     final_ans = data.decode('utf8')
            #     print(f"Final answer ->\n{final_ans}")
            #     logger.info(f"Final answer ->\n{final_ans}")
            #     agent_answer = final_ans
            #     end_event_received = True
            #     # End event indicates that the request finished successfully
            # elif 'trace' in event:
            #     logger.info(json.dumps(event['trace'], indent=2))
            # else:
            #     raise Exception("unexpected event.", event)
    # except Exception as e:
    #     raise Exception("unexpected event.", e)
    return {'text':agent_answer,'source':total_context,'last_tool':{'name':last_tool_name,'response':last_tool}}

        ####### Re-Rank ########
    
    #print("re-rank")
    
    # if(st.session_state.input_is_rerank == True and len(total_context)):
    #     ques = [{"question":question}]
    #     ans = [{"answer":total_context}]
        
    #     total_context = re_ranker.re_rank('rag','Cross Encoder',"",ques, ans)

    # llm_prompt = prompt_template.format(context=total_context[0],question=question)
    # output = invoke_models.invoke_llm_model( "\n\nHuman: {input}\n\nAssistant:".format(input=llm_prompt) ,False)
    # #print(output)
    # if(len(images_2)==0):
    #     images_2 = images
    # return {'text':output,'source':total_context,'image':images_2,'table':df}
