from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
f = open("/home/ubuntu/AI-search-with-amazon-opensearch-service/OpenSearchApp/utilities/colbert_vocab.txt", "r")
vocab = f.read()
vocab_arr = vocab.split("\n")
vocab_arr
vocab_dict={}
for index,n in enumerate(vocab_arr):
    vocab_dict[str(index)]=n
  


def vectorise(sentence,token_level_vectors):
    print("-------------colbert ---- 2-----------")
    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
     # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    if(token_level_vectors):
        return encoded_input['input_ids'][0].tolist(),model_output['last_hidden_state'][0]

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings[0].tolist()

def search(hits):
    print("-------------COLBERT------------4------------------------------------------")
    token_ids,token_vectors = vectorise(st.session_state.input_text,True)
    final_docs = []
    for ind,j in enumerate(hits):
        max_score_dict_list = []
        doc={"_source":
            {
            "description":j["_source"]["description"],"caption":j["_source"]["title"],
            "image_s3_url":j["_source"]["image_s3_url"],"price":j["_source"]["price"],
            "style":j["_source"]["style"],"category":j["_source"]["category"]},"_id":j["_id"],"_score":j["_score"]}
            
        if("gender_affinity" in j["_source"]):
            doc["_source"]["gender_affinity"] = j["_source"]["gender_affinity"]
        else:
            doc["_source"]["gender_affinity"] = ""
        #print(j["_source"]["title"])
        source_doc_token_keys = list(j["_source"].keys())
        with_s = [x for x in source_doc_token_keys if x.startswith("description-token-")]
        add_score = 0
        
        for index,i in enumerate(token_vectors):
            token = vocab_dict[str(token_ids[index])]
            if(token!='[SEP]' and token!='[CLS]'):
                query_token_vector = np.array(i)
                print("query token: "+token)
                print("-----------------")
                scores = []
                for m in with_s:
                    m_arr = m.split("-")
                    if(m_arr[-1]!='[SEP]' and m_arr[-1]!='[CLS]'):
                        #print("document token: "+m_arr[3])
                        doc_token_vector = np.array(j["_source"][m])
                        score = np.dot(query_token_vector,doc_token_vector)
                        scores.append({"doc_token":m_arr[3],"score":score})
                        #print({"doc_token":m_arr[3],"score":score})
                    
                newlist = sorted(scores, key=lambda d: d['score'], reverse=True)
                max_score = newlist[0]['score']
                add_score+=max_score
                max_score_dict_list.append(newlist[0])
                print(newlist[0])
        max_score_dict_list_sorted = sorted(max_score_dict_list, key=lambda d: d['score'], reverse=True)
        print(max_score_dict_list_sorted)
        # print(add_score)
        doc["total_score"] = add_score
        doc['max_score_dict_list_sorted'] = max_score_dict_list_sorted
        final_docs.append(doc)
    final_docs_sorted = sorted(final_docs, key=lambda d: d['total_score'], reverse=True)
    print("-------------COLBERT-----final--------")
    print(final_docs_sorted)
    return final_docs_sorted
        
                
        
                
        
