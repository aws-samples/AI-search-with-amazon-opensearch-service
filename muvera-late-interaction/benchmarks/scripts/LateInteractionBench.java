import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.analysis.standard.StandardAnalyzer;

import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * Standalone Lucene 10.5 micro-benchmark: native LateInteraction rescore (BinaryDocValues + SIMD)
 * vs the current OpenSearch approach (read multi-vectors from _source, deserialize, loop MaxSim).
 *
 * NOT an OpenSearch test — pure Lucene, single JVM. Isolates the rerank scoring kernel.
 *
 * Index each SciFact doc with:
 *   - a LateInteractionField  (native binary doc-values)   -> native path
 *   - a stored String field holding the same vectors as text -> _source-style path
 * Then, over a fixed candidate set, time:
 *   A) native  : LateInteractionFloatValuesSource (MaxSim, SIMD) reading binary doc-values
 *   B) _source : parse stored text -> float[][], manual MaxSim loop (mirrors Painless lateInteractionScore)
 *
 * Args: <doc_embeddings.json> <maxDocs> <candidates> <queryTokens> <iters>
 */
public class LateInteractionBench {
    static final int DIM = 128;

    public static void main(String[] args) throws Exception {
        String docFile   = args.length > 0 ? args[0] : "data_scifact/doc_embeddings.json";
        int maxDocs      = args.length > 1 ? Integer.parseInt(args[1]) : 2000;
        int candidates   = args.length > 2 ? Integer.parseInt(args[2]) : 40;
        int queryTokens  = args.length > 3 ? Integer.parseInt(args[3]) : 32;
        int iters        = args.length > 4 ? Integer.parseInt(args[4]) : 300;

        System.out.printf("LateInteractionBench: docs<=%d, candidates=%d, queryTokens=%d, iters=%d%n",
                maxDocs, candidates, queryTokens, iters);

        // ---- load doc multivectors (streaming-ish parse of the big JSON) ----
        List<float[][]> docs = loadDocs(docFile, maxDocs);
        System.out.printf("loaded %d docs, avg vecs=%.0f%n", docs.size(),
                docs.stream().mapToInt(d -> d.length).average().orElse(0));

        // ---- build a single Lucene index: LateInteractionField + stored text field ----
        Path idxDir = Files.createTempDirectory("libench");
        Directory dir = FSDirectory.open(idxDir);
        IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
        try (IndexWriter w = new IndexWriter(dir, cfg)) {
            for (float[][] mv : docs) {
                Document d = new Document();
                d.add(new LateInteractionField("li", mv));         // native binary doc-values
                d.add(new StoredField("src", encodeText(mv)));      // _source-style stored text
                w.addDocument(d);
            }
            w.commit();
        }
        DirectoryReader reader = DirectoryReader.open(dir);
        IndexSearcher searcher = new IndexSearcher(reader);

        // fixed candidate doc-id set (first `candidates` docs)
        int[] cand = new int[Math.min(candidates, docs.size())];
        for (int i = 0; i < cand.length; i++) cand[i] = i;

        // random query multivector (queryTokens x DIM), L2-ish normalized
        Random rng = new Random(42);
        float[][] query = randomMV(queryTokens, rng);

        // ---- warmup ----
        double warm = 0;
        for (int it = 0; it < 30; it++) {
            warm += scoreNative(searcher, reader, query, cand);
            warm += scoreSource(reader, query, cand);
        }
        if (Double.isNaN(warm)) System.out.println("warm nan");

        // ---- time NATIVE ----
        long t0 = System.nanoTime();
        double sinkN = 0;
        for (int it = 0; it < iters; it++) sinkN += scoreNative(searcher, reader, query, cand);
        long tN = System.nanoTime() - t0;

        // ---- time _SOURCE ----
        t0 = System.nanoTime();
        double sinkS = 0;
        for (int it = 0; it < iters; it++) sinkS += scoreSource(reader, query, cand);
        long tS = System.nanoTime() - t0;

        double nativeMsPerQ = tN / 1e6 / iters;
        double sourceMsPerQ = tS / 1e6 / iters;
        System.out.printf("%n=== RESULT (rerank %d candidates x %d query tokens, per query) ===%n", cand.length, queryTokens);
        System.out.printf("native  (LateInteractionField, binary DV + SIMD): %.3f ms/query%n", nativeMsPerQ);
        System.out.printf("_source (deserialize + manual MaxSim loop)      : %.3f ms/query%n", sourceMsPerQ);
        System.out.printf("SPEEDUP native vs _source: %.1fx%n", sourceMsPerQ / nativeMsPerQ);
        System.out.printf("(score check native=%.3f source=%.3f)%n", sinkN/iters, sinkS/iters);
        reader.close(); dir.close();
    }

    // ---- native scoring via LateInteractionFloatValuesSource (MaxSim over binary doc-values) ----
    static double scoreNative(IndexSearcher searcher, DirectoryReader reader, float[][] query, int[] cand) throws Exception {
        LateInteractionFloatValuesSource src =
            new LateInteractionFloatValuesSource("li", query, VectorSimilarityFunction.DOT_PRODUCT);
        DoubleValuesSource rewritten = src.rewrite(searcher);
        double total = 0;
        List<LeafReaderContext> leaves = reader.leaves();
        for (int docId : cand) {
            LeafReaderContext leaf = leaves.get(ReaderUtil.subIndex(docId, leaves));
            int leafDoc = docId - leaf.docBase;
            DoubleValues dv = rewritten.getValues(leaf, null);
            if (dv.advanceExact(leafDoc)) total += dv.doubleValue();
        }
        return total;
    }

    // ---- _source-style scoring: read stored text, parse to float[][], manual MaxSim ----
    static double scoreSource(DirectoryReader reader, float[][] query, int[] cand) throws Exception {
        StoredFields sf = reader.storedFields();
        double total = 0;
        for (int docId : cand) {
            Document d = sf.document(docId);
            float[][] dv = decodeText(d.get("src"));
            total += maxSim(query, dv);
        }
        return total;
    }

    static double maxSim(float[][] q, float[][] doc) {
        double sum = 0;
        for (float[] qt : q) {
            double best = -Double.MAX_VALUE;
            for (float[] dt : doc) {
                float s = VectorSimilarityFunction.DOT_PRODUCT.compare(qt, dt);
                if (s > best) best = s;
            }
            sum += best;
        }
        return sum;
    }

    // ---- helpers ----
    static float[][] randomMV(int n, Random rng) {
        float[][] mv = new float[n][DIM];
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int j = 0; j < DIM; j++) { mv[i][j] = (float) rng.nextGaussian(); norm += mv[i][j]*mv[i][j]; }
            float inv = (float)(1.0/Math.sqrt(norm));
            for (int j = 0; j < DIM; j++) mv[i][j] *= inv;
        }
        return mv;
    }

    static String encodeText(float[][] mv) {
        StringBuilder sb = new StringBuilder();
        for (float[] v : mv) { for (float x : v) { sb.append(x).append(' '); } sb.append('|'); }
        return sb.toString();
    }
    static float[][] decodeText(String s) {
        String[] rows = s.split("\\|");
        // last split may be empty
        int n = 0; for (String r : rows) if (!r.trim().isEmpty()) n++;
        float[][] mv = new float[n][];
        int idx = 0;
        for (String r : rows) {
            r = r.trim(); if (r.isEmpty()) continue;
            String[] toks = r.split("\\s+");
            float[] v = new float[toks.length];
            for (int j = 0; j < toks.length; j++) v[j] = Float.parseFloat(toks[j]);
            mv[idx++] = v;
        }
        return mv;
    }

    // streaming parse: scan the (huge) file char-by-char, extract each "embeddings":[[...]] block
    // by bracket-depth matching, stop after maxDocs. Avoids loading 2.8GB into memory.
    static List<float[][]> loadDocs(String file, int maxDocs) throws Exception {
        List<float[][]> out = new ArrayList<>();
        char[] key = "\"embeddings\"".toCharArray();
        try (java.io.BufferedReader br = Files.newBufferedReader(Paths.get(file))) {
            int c; int ki = 0;
            while ((c = br.read()) != -1) {
                if (c == key[ki]) { ki++; if (ki == key.length) {
                    // skip whitespace + ':' + whitespace to first '['
                    int ch; do { ch = br.read(); } while (ch != '[' && ch != -1);
                    if (ch == -1) break;
                    StringBuilder sb = new StringBuilder("[");
                    int depth = 1;
                    while (depth > 0 && (ch = br.read()) != -1) {
                        sb.append((char) ch);
                        if (ch == '[') depth++;
                        else if (ch == ']') depth--;
                    }
                    out.add(parseMV(sb.toString()));
                    if (out.size() >= maxDocs) break;
                    ki = 0;
                }} else { ki = (c == key[0]) ? 1 : 0; }
            }
        }
        return out;
    }
    static float[][] parseMV(String arr) {
        // arr = [[a,b,...],[c,d,...],...]
        List<float[]> vecs = new ArrayList<>();
        Matcher row = Pattern.compile("\\[([^\\[\\]]+)\\]").matcher(arr);
        while (row.find()) {
            String[] toks = row.group(1).split(",");
            float[] v = new float[toks.length];
            for (int j = 0; j < toks.length; j++) v[j] = Float.parseFloat(toks[j].trim());
            vecs.add(v);
        }
        return vecs.toArray(new float[0][]);
    }
}
