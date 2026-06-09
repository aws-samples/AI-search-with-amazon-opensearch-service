#!/bin/bash
cd /home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service
source /home/ec2-user/anaconda3/bin/activate python3
nohup streamlit run /home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service/OpenSearchApp/app.py --server.baseUrlPath="/proxy/absolute/8501" &
NOTEBOOK_NAME=$(jq -r '.ResourceDetail.NotebookInstanceName' /opt/ml/metadata/resource-metadata.json 2>/dev/null || echo "")
if [ -z "$NOTEBOOK_NAME" ]; then
  NOTEBOOK_NAME=$(aws sagemaker list-notebook-instances --query "NotebookInstances[?NotebookInstanceStatus=='InService'].NotebookInstanceName | [0]" --output text)
fi
echo "https:"$(echo $(echo $(aws sagemaker create-presigned-notebook-instance-url --notebook-instance-name $NOTEBOOK_NAME) | cut -d ':' -f 3) | cut -d '?' -f 1)"/proxy/absolute/8501"