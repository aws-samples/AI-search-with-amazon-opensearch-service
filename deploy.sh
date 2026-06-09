#!/bin/bash
cd /home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service
source /home/ec2-user/anaconda3/bin/activate python3

# Kill any existing streamlit process
pkill -f streamlit 2>/dev/null
sleep 2

# Launch the app
nohup streamlit run /home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service/OpenSearchApp/app.py --server.baseUrlPath="/proxy/absolute/8501" > /dev/null 2>&1 &

# Get the notebook instance name dynamically
NOTEBOOK_NAME=$(cat /opt/ml/metadata/resource-metadata.json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('ResourceName',''))" 2>/dev/null)
if [ -z "$NOTEBOOK_NAME" ]; then
  NOTEBOOK_NAME=$(aws sagemaker list-notebook-instances --query "NotebookInstances[?NotebookInstanceStatus=='InService'].NotebookInstanceName | [0]" --output text 2>/dev/null)
fi

if [ -z "$NOTEBOOK_NAME" ] || [ "$NOTEBOOK_NAME" = "None" ]; then
  echo "Could not determine notebook instance name. App is running at /proxy/absolute/8501"
else
  PRESIGNED_URL=$(aws sagemaker create-presigned-notebook-instance-url --notebook-instance-name "$NOTEBOOK_NAME" --output text 2>/dev/null)
  BASE_URL=$(echo "$PRESIGNED_URL" | cut -d '?' -f 1)
  echo "${BASE_URL}/proxy/absolute/8501"
fi