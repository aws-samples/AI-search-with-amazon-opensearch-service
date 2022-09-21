## Fine tuning BERT embeddings for Search Applications


Just like computer vision a few years ago, the decade-old field of natural language processing (NLP) is experiencing a fascinating renaissance. Not a month goes by without a new breakthrough! Indeed, thanks to the scalability and cost-efficiency of cloud-based infrastructure, researchers are finally able to train complex deep learning models on very large text datasets, in order to solve business problems such as question answering, sentence comparison, or text summarization.

In this respect, the Transformer  deep learning architecture has proven very successful, and has spawned several state of the art model families. One among them is Bidirectional Encoder Representations from Transformers (BERT): 340 million parameters [1]

As amazing as BERT is, training and optimizing it remains a challenging endeavor that requires a significant amount of time, resources, and skills, all the more when different languages are involved. Unfortunately, this complexity prevents most organizations from using these models effectively, if at all. Instead, wouldn’t it be great if we could just start from pre-trained versions and put them to work immediately? This is the exact challenge that Sentence Transformers (SBERT) and Hugging face frameworks are tackling.

![Higging Face, SBERT](/static/huggingfact-SBERT.jpeg)

With transformers, the “pretrain then fine-tune” recipe has emerged as the standard approach of applying BERT to specific downstream tasks such as classification, sequence labeling, and ranking. Typically, we start with a “base” pretrained transformer model such as the BERTBase and BERTLarge checkpoints directly downloadable from SBERT or the Hugging Face Transformers library. This model is then fine-tuned on task-specific labeled data drawn from the same distribution as the target task.

The SBERT framework is based on **PyTorch** and Transformers and offers a large collection of pre-trained models tuned for various tasks. We will be focussing on fine tuning the BERT model on data retrieval (search) usecase.

For data retrieval and ranking, the BERT model might be fine-tuned using a test collection comprised of queries and relevance judgments under a standard training, development (validation), and test split. The initial work is described in the paper Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks [2]. These fine-tuned embeddings can then be compared e.g. with cosine-similarity to find documents that match for a query that comes from a search engine. 




**Overview of Hands-on Labs**

1. Train a Sentence Transformer BERT model using Amazon SageMaker to fine tune the pre-trained BERT embeddings on data retrieval task
2. Deploy the fine-tuned BERT model on Amazon SageMaker for both real-time and batch inference.
3. Use Amazon OpenSearch service to index andd store the sentence embeddings and perform realtime search against query

**References**
- [1]  “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding“, Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.
- [2]  Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

