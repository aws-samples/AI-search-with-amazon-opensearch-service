## Steps to deploy the application

```
git clone -b Next-Gen-Search-Workshop https://github.com/aws-samples/AI-search-with-amazon-opensearch-service.git
wget https://d2d5zhnefzqxjo.cloudfront.net/neural-sparse-biencoder.tar.gz
aws s3 cp /home/ec2-user/neural-sparse-biencoder.tar.gz s3://${s3Bucket}
cd AI-search-with-amazon-opensearch-service
sudo chmod -R 0777 /home/ec2-user/
python3 -m venv /home/ec2-user/.myenv
source /home/ec2-user/.myenv/bin/activate
pip install streamlit boto3 requests_aws4auth opensearch-py torch==1.11.0 
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install -U sentence-transformers
pip install nltk ruamel_yaml langchain langchain-core langchain-community langchain-experimental ruamel_yaml lark
streamlit run OpenSearchApp/app.py
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

