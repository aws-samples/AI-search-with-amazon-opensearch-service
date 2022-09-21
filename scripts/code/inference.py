# Original Copyright 2022 The HuggingFace Team
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
    
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
import torch
import torch.nn.functional as F

# Helper: Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def model_fn(model_dir):
  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  model = BertModel.from_pretrained(model_dir)
  return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    
    # Tokenize sentences
    sentences = data.pop("inputs", data)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')["input_ids"]



    # Perform pooling
    #sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    #sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    # return dictonary, which will be json serializable
    return {"vectors": model(encoded_input)["pooler_output"].tolist()}
