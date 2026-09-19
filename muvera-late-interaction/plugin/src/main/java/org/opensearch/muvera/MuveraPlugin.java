/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.muvera;

import org.opensearch.ingest.Processor;
import org.opensearch.plugins.IngestPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.SearchPipelinePlugin;
import org.opensearch.search.pipeline.SearchRequestProcessor;

import java.util.Map;

/**
 * Standalone plugin that contributes the MUVERA fixed-dimensional-encoding
 * processors to OpenSearch through the public ingest and search-pipeline
 * extension points.
 *
 * <ul>
 *   <li>{@code muvera} — ingest processor: encodes multi-vectors into a single
 *       FDE vector stored in a {@code knn_vector} field for ANN prefetch.</li>
 *   <li>{@code muvera_query} — search request processor: encodes the query
 *       multi-vectors into an FDE and substitutes it into a template query.</li>
 * </ul>
 *
 * <p>The reranking step uses the {@code lateInteractionScore} Painless function,
 * which is provided by the k-NN plugin (bundled in the standard OpenSearch
 * distribution). This plugin therefore has a <em>runtime</em> dependency on the
 * k-NN plugin being installed, but no compile-time dependency on it.
 */
public class MuveraPlugin extends Plugin implements IngestPlugin, SearchPipelinePlugin {

    @Override
    public Map<String, Processor.Factory> getProcessors(Processor.Parameters parameters) {
        return Map.of(MuveraIngestProcessor.TYPE, new MuveraIngestProcessor.Factory());
    }

    @Override
    public Map<String, org.opensearch.search.pipeline.Processor.Factory<SearchRequestProcessor>> getRequestProcessors(
        SearchPipelinePlugin.Parameters parameters
    ) {
        return Map.of(MuveraSearchRequestProcessor.TYPE, new MuveraSearchRequestProcessor.Factory());
    }
}
