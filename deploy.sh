#!/bin/bash
cd /home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service
source /home/ec2-user/anaconda3/bin/activate python3
nohup streamlit run /home/ec2-user/SageMaker/AI-search-with-amazon-opensearch-service/OpenSearchApp/app.py --server.baseUrlPath="/proxy/absolute/8501" &
echo "https:"$(echo $(echo $(aws sagemaker create-presigned-notebook-instance-url --notebook-instance-name ml-search-opensearch) | cut -d ':' -f 3) | cut -d '?' -f 1)"/proxy/absolute/8501"