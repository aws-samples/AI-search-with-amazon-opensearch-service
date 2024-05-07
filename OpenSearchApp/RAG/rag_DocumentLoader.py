import boto3
import json
import os
import shutil
import time
from unstructured.partition.pdf import partition_pdf
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import streamlit as st
from PIL import Image 
import base64
import re
import torch
import base64
from anthropic import Anthropic
import requests 
from requests_aws4auth import AWS4Auth
import re_ranker
import utilities.invoke_models as invoke_models
import generate_csv_for_tables
from pdf2image import convert_from_bytes,convert_from_path
#import langchain

bedrock_runtime_client = boto3.client('bedrock-runtime',region_name='us-east-1')
textract_client = boto3.client('textract',region_name='us-east-1')

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



summary_prompt = """You are an assistant tasked with summarizing tables and text. \
Give a detailed summary of the table or text. Table or text chunk: {element} """








def generate_image_captions_(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds




def load_docs(inp):
    bucket = inp['bucket']
    key_file = inp['key']
    extracted_elements_list = []

    
    data_dir = '/home/ubuntu/pdfs'
    target_files = [os.path.join(data_dir,file_name) for file_name in os.listdir(data_dir)]
    
    

    Image.MAX_IMAGE_PIXELS = 100000000
    width = 2048
    height = 2048
    

    for target_file in target_files:
        tables_textract = generate_csv_for_tables.main_(target_file)
        #tables_textract = {}
        index_ = re.sub('[^A-Za-z0-9]+', '', (target_file.split("/")[-1].split(".")[0]).lower()) +"no_table"
        st.session_state.input_index = index_ 
        
        image_output_dir = '/home/ubuntu/figures/'+st.session_state.input_index+"/"
        image_output_dir_pdf = '/home/ubuntu/figures_pdf/'+st.session_state.input_index+"/"
        print(image_output_dir_pdf)
        if os.path.isdir(image_output_dir):
            shutil.rmtree(image_output_dir)
        if os.path.isdir(image_output_dir_pdf):
            shutil.rmtree(image_output_dir_pdf)

        path = os.path.join(image_output_dir, "") 
        os.mkdir(path)
        path_2 = os.path.join(image_output_dir_pdf, "") 
        os.mkdir(path_2)
        
        print("***")
        print(target_file)
        #image_output_dir_path = os.path.join(image_output_dir,target_file.split('/')[-1].split('.')[0])
        #os.mkdir(image_output_dir_path)
        
        # with open(target_file, "rb") as pdf_file:
        #     encoded_string_pdf = bytearray(pdf_file.read())
            
        #images_pdf = convert_from_path(target_file)
        
        # for index,image in enumerate(images_pdf):
        #     image.save(image_output_dir_pdf+"/"+st.session_state.input_index+"/"+str(index)+"_pdf.jpeg", 'JPEG')
        #     with open(image_output_dir_pdf+"/"+st.session_state.input_index+"/"+str(index)+"_pdf.jpeg", "rb") as read_img:
        #         input_encoded = base64.b64encode(read_img.read())
        # print(encoded_string_pdf)
        # tables_= textract_client.analyze_document( 
        #                                  Document={'Bytes': encoded_string_pdf},
        #                                  FeatureTypes=['TABLES']
        #                                 )
                                         
        # print(tables_)
        
        table_and_text_elements = partition_pdf(
            filename=target_file,
            extract_images_in_pdf=True,
            infer_table_structure=False,
            chunking_strategy="by_title", #Uses title elements to identify sections within the document for chunking
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir='/home/ubuntu/figures/'+st.session_state.input_index+'/',
        )
        tables = []
        texts = []
        print(table_and_text_elements)
        
        
        for table in tables_textract.keys():
            print(table)
            #print(tables_textract[table])
            tables.append({'table_name':table,'raw':tables_textract[table],'summary':invoke_models.invoke_llm_model(summary_prompt.format(element=tables_textract[table]),False)})
            time.sleep(4)
            
            
        for element in table_and_text_elements:
            # if "unstructured.documents.elements.Table" in str(type(element)):
            #     tables.append({'raw':str(element),'summary':invoke_models.invoke_llm_model(summary_prompt.format(element=str(element)),False)})
            #     tables_source.append({'raw':element,'summary':invoke_models.invoke_llm_model(summary_prompt.format(element=str(element)),False)})
           
            if "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))
        image_captions = {}


        for image_file in os.listdir(image_output_dir):
            print("image_processing")

            photo_full_path = image_output_dir+image_file
            photo_full_path_no_format = photo_full_path.replace('.jpg',"")
            
            with Image.open(photo_full_path) as image:
                image.verify()

            with Image.open(photo_full_path) as image:    
                
                file_type = 'jpg'
                path = image.filename.rsplit(".", 1)[0]
                image.thumbnail((width, height))
                image.save(photo_full_path_no_format+"-resized.jpg")
                
            with open(photo_full_path_no_format+"-resized.jpg", "rb") as read_img:
                input_encoded = base64.b64encode(read_img.read()).decode("utf8")
                

            image_captions[image_file] = {"caption":invoke_models.generate_image_captions_llm(input_encoded, "What's in this image?"),
                                          "encoding":input_encoded
                                        }
        print("image_processing done")
        #print(image_captions)

            #print(os.path.join('figures',image_file))
        extracted_elements_list = []
        extracted_elements_list.append({
                    'source': target_file,
                    'tables': tables,
                    'texts': texts,
                    'images': image_captions
                })
        documents = []
        documents_mm = []
        for extracted_element in extracted_elements_list:
            print("prepping data")
            texts = extracted_element['texts']
            tables = extracted_element['tables']
            images_data = extracted_element['images']
            src_doc = extracted_element['source']
            for text in texts:
                embedding = invoke_models.invoke_model(text)
                document = prep_document(text,text,'text',src_doc,'none',embedding)
                documents.append(document)
            for table in tables:
                table_raw = table['raw']
                
                
                table_summary = table['summary']
                embedding = invoke_models.invoke_model(table_summary)
                
                document = prep_document(table_raw,table_summary,'table*'+table['table_name'],src_doc,'none',embedding)
                documents.append(document)
            for file_name in images_data.keys():
                embedding = invoke_models.invoke_model_mm(image_captions[file_name]['caption'],image_captions[file_name]['encoding'])
                document = prep_document(image_captions[file_name]['caption'],image_captions[file_name]['caption'],'image_'+file_name,src_doc,image_captions[file_name]['encoding'],embedding)
                documents_mm.append(document)
                
                embedding = invoke_models.invoke_model(image_captions[file_name]['caption'])
                document = prep_document(image_captions[file_name]['caption'],image_captions[file_name]['caption'],'image_'+file_name,src_doc,'none',embedding)
                documents.append(document)

        
            
        os_ingest(index_, documents)
        os_ingest_mm(index_, documents_mm)

def prep_document(raw_element,processed_element,doc_type,src_doc,encoding,embedding):
    if('image' in doc_type):
        img_ = doc_type.split("_")[1]
    else:
        img_ = "None"
    document = { 
        "processed_element": processed_element,#re.sub(r"[^a-zA-Z0-9]+", ' ', processed_element) ,
        "raw_element_type": doc_type.split("*")[0],
        "raw_element": raw_element,#re.sub(r"[^a-zA-Z0-9]+", ' ', raw_element) ,
        "src_doc": src_doc.replace(","," "),
        "image": img_,
        
    }
    
    if(encoding!="none"):
        document["image_encoding"] = encoding
        document["processed_element_embedding_bedrock-multimodal"] = embedding
    else:
        document["processed_element_embedding"] = embedding
    
    if('table' in doc_type):
        document["table"] = doc_type.split("*")[1]
        
    return document



def os_ingest(index_,documents):
    print("ingesting data")
    #host = 'your collection id.region.aoss.amazonaws.com'
    if(ospy_client.indices.exists(index=index_)):
        ospy_client.indices.delete(index = index_)
    index_body = {
    "settings": {
        "index": {
            "knn": True,
            "default_pipeline": "rag-ingest-pipeline",
        "number_of_shards": 4
        }
    },
    "mappings": {
      "properties": {
        "processed_element": {
          "type": "text"
    },
             "raw_element": {
          "type": "text"
    },
        "processed_element_embedding": {
          "type": "knn_vector",
           "dimension":1536,
           "method": {
                  "engine": "faiss",
                  "space_type": "l2",
                  "name": "hnsw",
                  "parameters": {}
                }
    },
        # "processed_element_embedding_bedrock-multimodal": {
        #   "type": "knn_vector",
        #   "dimension": 1024,
        #   "method": {
        #     "engine": "faiss",
        #     "space_type": "l2",
        #     "name": "hnsw",
        #     "parameters": {}
        #   }
        # },
        #  "image_encoding": {
        #   "type": "binary"
        # },
    "raw_element_type": {
          "type": "text"
    },
   "processed_element_embedding_sparse": {
          "type": "rank_features"
        },
    "src_doc": {
          "type": "text"
    },
    "image":{ "type": "text"}
    
    }
    }
    }
    response = ospy_client.indices.create(index_, body=index_body)

    for doc in documents:
        print("----------doc------------")
        if(doc['image']!='None'):
            print("image insert")
            print(doc['image'])
        
        response = ospy_client.index(
            index = index_,
            body = doc,
        )


def os_ingest_mm(index_,documents_mm):
    #host = 'your collection id.region.aoss.amazonaws.com'
    index_ = index_+"_mm"
    if(ospy_client.indices.exists(index=index_)):
        ospy_client.indices.delete(index = index_)
    index_body = {
    "settings": {
        "index": {
            "knn": True,
           # "default_pipeline": "rag-ingest-pipeline",
        "number_of_shards": 4
        }
    },
    "mappings": {
      "properties": {
        "processed_element": {
          "type": "text"
    },
             "raw_element": {
          "type": "text"
    },
      
        "processed_element_embedding_bedrock-multimodal": {
          "type": "knn_vector",
          "dimension": 1024,
          "method": {
            "engine": "faiss",
            "space_type": "l2",
            "name": "hnsw",
            "parameters": {}
          }
        },
         "image_encoding": {
          "type": "binary"
        },
    "raw_element_type": {
          "type": "text"
    },
  
    "src_doc": {
          "type": "text"
    },
    "image":{ "type": "text"}
    
    }
    }
    }
    response = ospy_client.indices.create(index_, body=index_body)

    for doc in documents_mm:
        #print("----------doc------------")
        #print(doc)
        
        response = ospy_client.index(
            index = index_,
            body = doc,
        )




