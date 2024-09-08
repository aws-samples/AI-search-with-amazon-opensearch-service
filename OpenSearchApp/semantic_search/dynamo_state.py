import streamlit as st
import boto3
import subprocess



def get_region():
    # Define the command
    command = "ec2-metadata --availability-zone | sed 's/.$//'"

    # Run the command and capture the output
    output = subprocess.check_output(command, shell=True, text=True)

    # Print the command output
    print("Command output:")
    st.session_state.REGION = output.split(":")[1].strip()

    return st.session_state.REGION



def store_in_dynamo(key,val):
    dynamo_client = boto3.client('dynamodb',region_name=st.session_state.REGION)
    response = dynamo_client.put_item(
    Item={
        'store_key': {
            'S': key,
    },
             'store_val': {
            'S': val,
    }},
    TableName='dynamo_store_key_value',
)
    
def get_from_dynamo(key):
    dynamo_client = boto3.client('dynamodb',region_name=st.session_state.REGION)
    res = dynamo_client.get_item( TableName='dynamo_store_key_value',Key = {'store_key': {'S': key}})
    if('Item' not in res):
        return ""
    else:
        return res['Item']['store_val']['S']

