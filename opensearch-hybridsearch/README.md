## Hybrid Search with OpenSearch

Traditional keyword-based search algorithm which works by matching user queries with specific words or phrases is more suitable for exact matches, provides good interpretability of results, requires no fine-tuning and generalises well across domains and importantly it is fast in fetching documents. However, this approach may sometimes lead to imprecise results, especially when dealing with ambiguous terms or variations in language due to poor contextual understanding.

On the other hand, vector search enables a more context-aware search understanding the natural language questions of users. However, this method requires fine-tuning of the ML model for the involved domain, provides poor interpretability of results and importantly requires more memory resources yet not faster compared to basic keyword search.

It will be beneficial to leverage the best features of each algorithm to complement the limitations of the other using a hybrid search system. Such integration can lead to improved accuracy in understanding user intent and delivering relevant results that are contextually appropriate. In this section you will learn about implementing such a hybrid search system using Amazon OpenSearch service to combine keyword and vector search.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

