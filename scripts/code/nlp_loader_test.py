#!/usr/bin/env python
# coding: utf-8

# Original Copyright 2019 Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
    
import os
import subprocess
import sys

# Install dependencies
def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
install('sentence_transformers')



import boto3
import argparse
import gzip
import logging
from itertools import islice
import math

import json
from torch.utils.data import DataLoader
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle


# os.environ["SM_MODEL_DIR"] = "/tmp/model"
# os.environ["SM_CHANNEL_TRAINING"] = "/tmp/data"
# os.environ["SM_CHANNEL_TESTING"] = "/tmp/data"
# os.environ["SM_HOSTS"] = '["algo-1"]'
# os.environ["SM_CURRENT_HOST"] = "algo-1"
# os.environ["SM_NUM_GPUS"] = "0"




from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def train(args):

    # The  model we want to fine-tune
    model_name = 'distilbert-base-uncased'

    train_batch_size = 64           #Increasing the train batch size improves the model performance, but requires more GPU memory
    max_seq_length = 300            #Max length for passages. Increasing it, requires more GPU memory
    ce_score_margin = 3.0             #Margin for the CrossEncoder score between negative and positive passages
    num_negs_per_system = 5         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
    num_epochs = 10                 # Number of epochs we want to train

    # Load our embedding model
    if False:
        logging.info("use pretrained SBERT model")
        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
    else:
        logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    model_save_path = 'output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


    # In[5]:


    ### Now we read the MS Marco dataset
    data_folder = 'msmarco-data'

    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    logging.info("Read corpus: collection.tsv")
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage


    # In[6]:


    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    # In[7]:


    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query




    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)

    # As training data we use hard-negatives that have been mined using various systems
    hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
    # if not os.path.exists(hard_negatives_filepath):
    #     logging.info("Download cross-encoder scores file")
    #     util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)


    # In[12]:



    train_queries = {}


    def take_(n, iterable):
        "Return first n items of the iterable as a list"
        return dict(islice(iterable, n))

    n_items = take_(1000, ce_scores.items())

    def f(x):
        return math.trunc(x)

    for i,j in n_items.items():
        dict_ = {k: f(v) for k, v in n_items[i].items()}
        dict__ = dict(sorted(dict_.items(), key=lambda item: item[1],reverse=True))
        result = {}

        for key,value in dict__.items():
            if value not in result.values():
                result[key] = value

        n_items__ = take_(5, result.items())
        #print(n_items__)
        n_items___={}
        temp_arr=[]

        n_items___['qid'] = i
        n_items___['query'] = queries[i]
        firstKey = next(iter(n_items__))
        temp_arr.append(firstKey)
        n_items___['pos'] = temp_arr
        del n_items__[firstKey]
        n_items___['neg'] = set(n_items__.keys())

        train_queries[i]=n_items___

    


    
    del ce_scores

    logging.info("Train queries: {}".format(len(train_queries)))


    # We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
    # on-the-fly based on the information from the mined-hard-negatives jsonl file.
    class MSMARCODataset(Dataset):
        def __init__(self, queries, corpus):
            self.queries = queries
            self.queries_ids = list(queries.keys())
            self.corpus = corpus

            for qid in self.queries:
                self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
                self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
                random.shuffle(self.queries[qid]['neg'])

        def __getitem__(self, item):
            query = self.queries[self.queries_ids[item]]
            query_text = query['query']

            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)

            neg_id = query['neg'].pop(0)    #Pop negative and add at end
            neg_text = self.corpus[neg_id]
            query['neg'].append(neg_id)

            return InputExample(texts=[query_text, pos_text, neg_text])

        def __len__(self):
            return len(self.queries)




    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataset = MSMARCODataset(train_queries, corpus=corpus)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)




    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              warmup_steps=10,
              use_amp=True,
              checkpoint_path=model_save_path,
              checkpoint_save_steps=len(train_dataloader),
              optimizer_params = {'lr': 0.00002}
              )



    model.save(args.model_dir)

def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )


    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

