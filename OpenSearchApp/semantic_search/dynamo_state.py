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

dynamo_client = boto3.client('dynamodb',region_name=get_region())


def store_in_dynamo(key,val):
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
    res = dynamo_client.get_item( TableName='dynamo_store_key_value',Key = {'store_key': {'S': key}})
    if('Item' not in res):
        return ""
    else:
        return res['Item']['store_val']['S']
    
def update_in_dynamo(key,attr_name,attr_val):
    dynamo_client.update_item( TableName='dynamo_store_key_value',
                                          ExpressionAttributeNames={
                                                '#Y': attr_name,
                                            },
                                            ExpressionAttributeValues={
                                                
                                                ':y': {
                                                    'S': attr_val,
                                                },
                                            },Key = {'store_key': {'S': key}},
                                                ReturnValues='ALL_NEW',
                                                UpdateExpression='SET #Y = :y',
                                            )

