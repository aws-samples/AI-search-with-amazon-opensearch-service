import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from transformers import AutoProcessor
from PIL import Image
from io import BytesIO
import streamlit as st
import requests
from pdf2image import convert_from_path
from pypdf import PdfReader
import numpy as np
import re
import base64
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from PIL import Image 
import requests 
from requests_aws4auth import AWS4Auth
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
from colpali_engine.utils.image_utils import scale_image, get_base64_image
import os
parent_dirname = "/".join((os.path.dirname(__file__)).split("/")[0:-1])


if torch.cuda.is_available():
    device = torch.device("cuda")
    type = torch.bfloat16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    type = torch.float32
else:
    device = torch.device("cpu")
    type = torch.float32
    
region = 'us-east-1'
service = 'es'

credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

ospy_client = OpenSearch(
    hosts = [{'host': 'search-opensearchservi-75ucark0bqob-bzk6r6h2t33dlnpgx2pdeg22gi.us-east-1.es.amazonaws.com', 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)
 
model_name = "vidore/colpali-v1.2"
model = ColPali.from_pretrained(
    "vidore/colpaligemma-3b-pt-448-base", torch_dtype=type
).eval()
model.load_adapter(model_name)
model = model.eval()
model.to(device)
processor = AutoProcessor.from_pretrained(model_name)

# def download_pdf(url):
#     response = requests.get(url)
#     if response.status_code == 200:
#         return BytesIO(response.content)
#     else:
#         raise Exception(f"Failed to download PDF: Status code {response.status_code}")


def get_pdf_images(target_file):
    reader = PdfReader(target_file)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    images = convert_from_path(target_file)
    assert len(images) == len(page_texts)
    return (images, page_texts)   
    
def process_doc(inp):#   need to change for many files as input upload
    
    print("input_doc")
    print(inp)
    extracted_elements_list = []
    data_dir = parent_dirname+"/pdfs"
    target_files = [os.path.join(data_dir,inp["key"])] ##### need to change for multiple input pdf files
    pdf_name = inp["key"]
    for pdf in target_files:
        index_ = re.sub('[^A-Za-z0-9]+', '', (pdf.split("/")[-1].split(".")[0]).lower())
        st.session_state.input_index = index_
        pdf_pages = {index_:{}}
        page_images, page_texts = get_pdf_images(pdf) 
        pdf_pages[index_]["images"] = page_images
        pdf_pages[index_]["texts"] = page_texts
        page_embeddings = []
        ######## retrieve embeddings for page images #########
        dataloader = DataLoader(
        pdf_pages[index_]["images"],
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
        )
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
                page_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        pdf_pages[index_]["embeddings"] = page_embeddings
        create_index(index_)
        prep_ingest_page_docs(pdf_pages[index_],index_)
        
def create_index(index_):
  print("creating index")
  if(ospy_client.indices.exists(index=index_)):
        ospy_client.indices.delete(index = index_)
  payload = {
  # "settings": {
  #   "index.knn": True
  # },
  "mappings": {
    "_source": {
      "excludes": [
        "page_image"
      ]
    },
    "properties": {
    "pdf_url": {
       "type": "text"
        },
        "page_embedding": {
          "type": "knn_vector",
          "dimension": 128,
          "method": {
            "engine": "faiss",
            "space_type": "l2",
            "name": "hnsw",
            "parameters": {}
          }
        },
        "pdf_title": {
          "type": "text"
        },
        "page_image":{"type": "binary"},
        "page_texts": {
          "type": "text"
        },
        "page_num": {
          "type": "integer"
        }
        }}
        }
  response = ospy_client.indices.create(index_, body=payload)
            
def prep_ingest_page_docs(pdf_,pdf_name):
    page_wise_documents = []
    images_base_64=[]
    for page_num, image in enumerate(pdf_["images"]): #iterate through pages
        #images_base_64.append(get_base64_image(image, add_url_prefix=False))
        
        #### storing the page images in local
        imgstring = get_base64_image(image, add_url_prefix=False)
        imgdata = base64.b64decode(imgstring)
        image_output_dir = parent_dirname+'/figures/'+st.session_state.input_index+'/'
        if os.path.isdir(image_output_dir):
            shutil.rmtree(image_output_dir)
        os.mkdir(image_output_dir)
        filename = parent_dirname+'/figures/'+st.session_state.input_index+'/'+pdf_name+'_'+str(page_num)+'.jpg'  
        with open(filename, 'wb') as f:
            f.write(imgdata)
        ####
        patch_vectors = pdf_['embeddings'][page_num]
        doc = {
        
            "pdf_url": pdf_["url"],
            "pdf_title": pdf_["title"],
            #"page_image": get_base64_image(image, add_url_prefix=False),
            "page_num": page_num,
            "page_texts": pdf_["texts"][page_num],  # Array of text per page
            "page_embedding": (np.average(patch_vectors, axis=0)).tolist()
            }
        for patch_num,patch in enumerate(patch_vectors):
            doc['patch-'+str(patch_num)] = patch.tolist()
            
        response = ospy_client.index(
            index = pdf_name,
            body = doc,
        )
        print(response)
        page_wise_documents.append(doc)
        
#def ingest_page_docs():
  
  
    