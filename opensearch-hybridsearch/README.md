**AWS Big Data Blog**

# Hybrid Search with Amazon OpenSearch Service

by Hajer Bouafif and Praveen Mohan Prasad | on 19 MAR 2024 | in Advanced (300), Amazon OpenSearch Service,
Amazon SageMaker | Permalink |  Comments |  Share
Amazon OpenSearch Service has been a long-standing supporter of both lexical and semantic search, facilitated
by its utilization of the k-nearest neighbors (k-NN) plugin. By using OpenSearch Service as a vector database, you
can seamlessly combine the advantages of both lexical and vector search. The introduction of the neural search
feature in OpenSearch Service 2.9 further simplifie s integration with artific ial intelligence (AI) and machine
learning (ML) models, facilitating the implementation of semantic search.

Lexical search using TF/IDF or BM25 has been the workhorse of search systems for decades. These traditional
lexical search algorithms match user queries with exact words or phrases in your documents. Lexical search is
more suitable for exact matches, provides low latency, and offers good interpretability of results and generalizes
well across domains. However, this approach does not consider the context or meaning of the words, which can
lead to irrelevant results.

In the past few years, semantic search methods based on vector embeddings have become increasingly popular
to enhance search. Semantic search enables a more context-aware search, understanding the natural language
questions of user queries. However, semantic search powered by vector embeddings requires fine-tuning of the
ML model for the associated domain (such as healthcare or retail) and more memory resources compared to basic
lexical search.

Both lexical search and semantic search have their own strengths and weaknesses. Combining lexical and vector
search improves the quality of search results by using their best features in a hybrid model. OpenSearch Service
2.11 now supports out-of-the-box hybrid query capabilities that make it straightforward for you to implement a
hybrid search model combining lexical search and semantic search.

This post explains the internals of hybrid search and how to build a hybrid search solution using OpenSearch
Service. We experiment with sample queries to explore and compare lexical, semantic, and hybrid search. All the
code used in this post is publicly available in the GitHub repository.

## Hybrid search with OpenSearch Service

In general, hybrid search to combine lexical and semantic search involves the following steps:

1. Run a semantic and lexical search using a compound search query clause.
2. Each query type provides scores on different scales. For example, a Lucene lexical search query will return a
    score between 1 and infin ity. On the other hand, a semantic query using the Faiss engine returns scores
    between 0 and 1. Therefore, you need to normalize the scores coming from each type of query to put them
    on the same scale before combining the scores. In a distributed search engine, this normalization needs to
    happen at the global level rather than shard or node level.
3. After the scores are all on the same scale, they’re combined for every document.
4. Reorder the documents based on the new combined score and render the documents as a response to the
    query.


Prior to OpenSearch Service 2.11, search practitioners would need to use compound query types to combine
lexical and semantic search queries. However, this approach does not address the challenge of global
normalization of scores as mentioned in Step 2.

OpenSearch Service 2.11 added the support of hybrid query by introducing the score normalization processor in
search pipelines. Search pipelines take away the heavy lifting of building normalization of score results and
combination outside your OpenSearch Service domain. Search pipelines run inside the OpenSearch Service
domain and support three types of processors: search request processor, search response processor, and search
phase results processor.

In a hybrid search, the search phase results processor runs between the query phase and fetch phase at the
coordinator node (global) level. The following diagram illustrates this workflow.

The hybrid search workflow in OpenSearch Service contains the following phases:

```
Query phase – The firs t phase of a search request is the query phase, where each shard in your index runs the
search query locally and returns the document ID matching the search request with relevance scores for each
document.
Score normalization and combination – The search phase results processor runs between the query phase
and fetch phase. It uses the normalization processer to normalize scoring results from BM25 and KNN
subqueries. The search processor supports min_max and L2-Euclidean distance normalization methods. The
processor combines all scores, compiles the fin al list of ranked document IDs, and passes them to the fetch
phase. The processor supports arithmetic_mean, geometric_mean, and harmonic_mean to combine scores.
```

```
Fetch phase – The fin al phase is the fetch phase, where the coordinator node retrieves the documents that
matches the fin al ranked list and returns the search query result.
```
## Solution overview

In this post, you build a web application where you can search through a sample image dataset in the retail
space, using a hybrid search system powered by OpenSearch Service. Let’s assume that the web application is a
retail shop and you as a consumer need to run queries to search for women’s shoes.

For a hybrid search, you combine a lexical and semantic search query against the text captions of images in the
dataset. The end-to-end search application high-level architecture is shown in the following figure.

The workflow contains the following steps:

1. You use an Amazon SageMaker notebook to index image captions and image URLs from the Amazon Berkeley
    Objects Dataset stored in Amazon Simple Storage Service (Amazon S3) into OpenSearch Service using the
    OpenSearch ingest pipeline. This dataset is a collection of 147,702 product listings with multilingual
    metadata and 398,212 unique catalog images. You only use the item images and item names in US English.
    For demo purposes, you use approximately 1,600 products.
2. OpenSearch Service calls the embedding model hosted in SageMaker to generate vector embeddings for the
    image caption. You use the GPT-J-6B variant embedding model, which generates 4,096 dimensional vectors.
3. Now you can enter your search query in the web application hosted on an Amazon Elastic Compute Cloud
    (Amazon EC2) instance (c5.large). The application client triggers the hybrid query in OpenSearch Service.
4. OpenSearch Service calls the SageMaker embedding model to generate vector embeddings for the search
    query.
5. OpenSearch Service runs the hybrid query, combines the semantic search and lexical search scores for the
    documents, and sends back the search results to the EC2 application client.


Let’s look at Steps 1, 2, 4, and 5 in more detail.

### Step 1: Ingest the data into OpenSearch

In Step 1, you create an ingest pipeline in OpenSearch Service using the text_embedding processor to generate
vector embeddings for the image captions.

After you defin e a k-NN index with the ingest pipeline, you run a bulk index operation to store your data into the
k-NN index. In this solution, you only index the image URLs, text captions, and caption embeddings where the
fie ld type for the caption embeddings is k-NN vector.

### Step 2 and Step 4: OpenSearch Service calls the SageMaker embedding model

In these steps, OpenSearch Service uses the SageMaker ML connector to generate the embeddings for the image
captions and query. The blue box in the preceding architecture diagram refers to the integration of OpenSearch
Service with SageMaker using the ML connector feature of OpenSearch. This feature is available in OpenSearch
Service starting from version 2.9. It enables you to create integrations with other ML services, such as SageMaker.

### Step 5: OpenSearch Service runs the hybrid search query

OpenSearch Service uses the search phase results processor to perform a hybrid search. For hybrid scoring,
OpenSearch Service uses the normalization, combination, and weights config uration settings that are set in the
normalization processor of the search pipeline.

## Prerequisites

Before you deploy the solution, make sure you have the following prerequisites:

```
An AWS account
Familiarity with the Python programming language
Familiarity with AWS Identity and Access Management (IAM), Amazon EC2, OpenSearch Service, and
SageMaker
```
## Deploy the hybrid search application to your AWS account

To deploy your resources, use the provided AWS CloudFormation template. Supported AWS Regions are us-east-
1, us-west-2, and eu-west-1. Complete the following steps to launch the stack:

1. On the AWS CloudFormation console, create a new stack.
2. For **Template source** , select **Amazon S3 URL**.
3. For **Amazon S3 URL** , enter the path for the template for deploying hybrid search.
4. Choose **Next**.


5. Name the stack hybridsearch.
6. Keep the remaining settings as default and choose **Submit**.
7. The template stack should take 15 minutes to deploy. When it’s done, the stack status will show as
    **CREATE_COMPLETE**.
8. When the stack is complete, navigate to the stack **Outputs** tab.
9. Choose the SagemakerNotebookURL link to open the SageMaker notebook in a separate tab.


10. In the SageMaker notebook, navigate to the AI-search-with-amazon-opensearch-service/
    opensearch-hybridsearch directory and open HybridSearch.ipynb.


11. If the notebook prompts to set the kernel, Choose the conda_pytorch_p310 kernel from the drop-down
    menu, then choose **Set Kernel**.
12. The notebook should look like the following screenshot.


Now that the notebook is ready to use, follow the step-by-step instructions in the notebook. With these steps,
you create an OpenSearch SageMaker ML connector and a k-NN index, ingest the dataset into an OpenSearch
Service domain, and host the web search application on Amazon EC2.

## Run a hybrid search using the web application

The web application is now deployed in your account and you can access the application using the URL generated
at the end of the SageMaker notebook.

Copy the generated URL and enter it in your browser to launch the application.

Complete the following steps to run a hybrid search:


1. Use the search bar to enter your search query.
2. Use the drop-down menu to select the search type. The available options are **Keyword Search** , **Vector**
    **Search** , and **Hybrid Search**.
3. Choose **GO** to render results for your query or regenerate results based on your new settings.
4. Use the left pane to tune your hybrid search config uration:
    Under **Weight for Semantic Search** , adjust the slider to choose the weight for semantic subquery. Be
    aware that the total weight for both lexical and semantic queries should be 1.0. The closer the weight is to
    1.0, the more weight is given to the semantic subquery, and this setting minus 1.0 goes as weightage to
    the lexical query.
    For **Select the normalization type** , choose the normalization technique (min_max or L2).
    For **Select the Score Combination type** , choose the score combination techniques: arithmetic_mean,
    geometric_mean, or harmonic_mean.

## Experiment with Hybrid Search

In this post, you run four experiments to understand the differences between the outputs of each search type.

As a customer of this retail shop, you are looking for women’s shoes, and you don’t know yet what style of shoes
you would like to purchase. You expect that the retail shop should be able to help you decide according to the
following parameters:

```
Not to deviate from the primary attributes of what you search for.
Provide versatile options and styles to help you understand your preference of style and then choose one.
```
As your firs t step, enter the search query “women shoes” and choose **5** as the number of documents to output.

Next, run the following experiments and review the observation for each search type

### Experiment 1: Lexical search


For a lexical search, choose **Keyword Search** as your search type, then choose **GO**.

The keyword search runs a lexical query, looking for same words between the query and image captions. In the
firs t four results, two are women’s boat-style shoes identifie d by common words like “women” and “shoes.” The
other two are men’s shoes, linked by the common term “shoes.” The last result is of style “sandals,” and it’s
identifie d based on the common term “shoes.”

In this experiment, the keyword search provided three relevant results out of fiv e—it doesn’t completely capture
the user’s intention to have shoes only for women.

### Experiment 2: Semantic search

For a semantic search, choose **Semantic search** as the search type, then choose **GO**.


The semantic search provided results that all belong to one particular style of shoes, “boots.” Even though the
term “boots” was not part of the search query, the semantic search understands that terms “shoes” and “boots”
are similar because they are found to be nearest neighbors in the vector space.

In this experiment, when the user didn’t mention any specific shoe styles like boots, the results limited the user’s
choices to a single style. This hindered the user’s ability to explore a variety of styles and make a more informed
decision on their preferred style of shoes to purchase.

Let’s see how hybrid search can help in this use case.

### Experiment 3: Hybrid search

Choose **Hybrid Search** as the search type, then choose **GO**.

In this example, the hybrid search uses both lexical and semantic search queries. The results show two “boat
shoes” and three “boots,” reflecting a blend of both lexical and semantic search outcomes.

In the top two results, “boat shoes” directly matched the user’s query and were obtained through lexical search.
In the lower-ranked items, “boots” was identifie d through semantic search.

In this experiment, the hybrid search gave equal weighs to both lexical and semantic search, which allowed users
to quickly fin d what they were looking for (shoes) while also presenting additional styles (boots) for them to
consider.

### Experiment 4: Fine-tune the hybrid search configuration


In this experiment, set the weight of the vector subquery to 0.8, which means the keyword search query has a
weightage of 0.2. Keep the normalization and score combination settings set to default. Then choose **GO** to
generate new results for the preceding query.

Providing more weight to the semantic search subquery resulted in higher scores to the semantic search query
results. You can see a similar outcome as the semantic search results from the second experiment, with five
images of boots for women.

You can further fin e-tune the hybrid search results by adjusting the combination and normalization techniques.

In a benchmark conducted by the OpenSearch team using publicly available datasets such as BEIR and Amazon
ESCI, they concluded that the min_max normalization technique combined with the arithmetic_mean
score combination technique provides the best results in a hybrid search.

You need to thoroughly test the different fin e-tuning options to choose what is the most relevant to your
business requirements.

## Overall observations

From all the previous experiments, we can conclude that the hybrid search in the third experiment had a
combination of results that looks relevant to the user in terms of giving exact matches and also additional styles
to choose from. The hybrid search matches the expectation of the retail shop customer.

## Clean up

To avoid incurring continued AWS usage charges, make sure you delete all the resources you created as part of
this post.

To clean up your resources, make sure you delete the S3 bucket you created within the application before you
delete the CloudFormation stack.


