package semantic;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.ResourceIterator;

/**
 * Helper class for semantic graph operations.
 * Core innovation: cosine similarity between query embedding and stored embeddings.
 */
public class SemanticGraphHelper {

    // ── Cosine Similarity ─────────────────────────────────────────────────
    /**
     * Computes cosine similarity between two vectors.
     * Returns value in [-1, 1], higher = more similar.
     */
    public static double cosineSimilarity(List<Double> vecA, double[] vecB) {
        if (vecA == null || vecB == null || vecA.size() != vecB.length) {
            return 0.0;
        }
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < vecA.size(); i++) {
            double a = vecA.get(i);
            double b = vecB[i];
            dot   += a * b;
            normA += a * a;
            normB += b * b;
        }
        if (normA == 0.0 || normB == 0.0) return 0.0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    // ── Convert Neo4j stored embedding to double[] ─────────────────────
    /**
     * Neo4j stores float arrays as double[] or long[].
     * This method handles both cases safely.
     */
    public static double[] toDoubleArray(Object rawEmbedding) {
        if (rawEmbedding instanceof double[]) {
            return (double[]) rawEmbedding;
        }
        if (rawEmbedding instanceof float[]) {
            float[] fa = (float[]) rawEmbedding;
            double[] da = new double[fa.length];
            for (int i = 0; i < fa.length; i++) da[i] = fa[i];
            return da;
        }
        if (rawEmbedding instanceof long[]) {
            long[] la = (long[]) rawEmbedding;
            double[] da = new double[la.length];
            for (int i = 0; i < la.length; i++) da[i] = (double) la[i];
            return da;
        }
        return null;
    }

    // ── matchVertex ───────────────────────────────────────────────────────
    /**
     * Find top-K nodes whose 'embedding' property is most similar
     * to the given queryEmbedding.
     *
     * Node schema expected:
     *   (:Entity {mid: "m.0m_sb", name: "Northern line", embedding: [...]})
     *
     * @param tx             current transaction
     * @param queryEmbedding the query vector (from Python sentence-transformer)
     * @param topK           number of results to return
     * @return               ranked list of ScoredNode
     */
    public static List<ScoredNode> matchVertex(
            Transaction tx,
            List<Double> queryEmbedding,
            long topK) {

        List<ScoredNode> results = new ArrayList<>();

        // Use Neo4j vector index if available (Neo4j 5.x)
        // Falls back to linear scan if index not present
        try {
            String cypher =
                "CALL db.index.vector.queryNodes('entity_index', $k, $emb) " +
                "YIELD node, score " +
                "RETURN node, score";
            tx.execute(cypher, java.util.Map.of(
                "k", (int) topK,
                "emb", queryEmbedding
            )).forEachRemaining(row -> {
                Node node = (Node) row.get("node");
                double score = ((Number) row.get("score")).doubleValue();
                results.add(new ScoredNode(node, score));
            });
            if (!results.isEmpty()) return results;
        } catch (Exception e) {
            // Vector index not available, fall back to linear scan
        }

        // Linear scan fallback
        try (ResourceIterator<Node> nodes = tx.findNodes(
                org.neo4j.graphdb.Label.label("Entity"))) {
            while (nodes.hasNext()) {
                Node node = nodes.next();
                if (node.hasProperty("embedding")) {
                    double[] stored = toDoubleArray(node.getProperty("embedding"));
                    if (stored != null) {
                        double score = cosineSimilarity(queryEmbedding, stored);
                        results.add(new ScoredNode(node, score));
                    }
                }
            }
        }

        results.sort(Comparator.comparingDouble(ScoredNode::getScore).reversed());
        return results.subList(0, (int) Math.min(topK, results.size()));
    }

    // ── matchRelation ─────────────────────────────────────────────────────
    /**
     * Find top-K Relation nodes whose 'embedding' is most similar
     * to the given queryEmbedding.
     *
     * Relation node schema:
     *   (:Relation {name: "transport.transit_line.terminus", embedding: [...]})
     *
     * This is the KEY method for solving Schema Drift:
     * Instead of exact relation name matching (BM25), we use vector similarity.
     *
     * @param tx             current transaction
     * @param queryEmbedding the query vector encoding the question/relation mention
     * @param topK           number of results to return
     * @return               ranked list of ScoredRelation
     */
    public static List<ScoredRelation> matchRelation(
            Transaction tx,
            List<Double> queryEmbedding,
            long topK) {

        List<ScoredRelation> results = new ArrayList<>();

        // Try Neo4j vector index first
        try {
            String cypher =
                "CALL db.index.vector.queryNodes('relation_index', $k, $emb) " +
                "YIELD node, score " +
                "RETURN node.name AS name, score";
            tx.execute(cypher, java.util.Map.of(
                "k", (int) topK,
                "emb", queryEmbedding
            )).forEachRemaining(row -> {
                String name  = (String) row.get("name");
                double score = ((Number) row.get("score")).doubleValue();
                results.add(new ScoredRelation(name, score));
            });
            if (!results.isEmpty()) return results;
        } catch (Exception e) {
            // Vector index not available, fall back to linear scan
        }

        // Linear scan fallback over :Relation nodes
        try (ResourceIterator<Node> nodes = tx.findNodes(
                org.neo4j.graphdb.Label.label("Relation"))) {
            while (nodes.hasNext()) {
                Node node = nodes.next();
                if (node.hasProperty("embedding") && node.hasProperty("name")) {
                    double[] stored = toDoubleArray(node.getProperty("embedding"));
                    if (stored != null) {
                        double score = cosineSimilarity(queryEmbedding, stored);
                        String name  = (String) node.getProperty("name");
                        results.add(new ScoredRelation(name, score));
                    }
                }
            }
        }

        results.sort(Comparator.comparingDouble(ScoredRelation::getScore).reversed());
        return results.subList(0, (int) Math.min(topK, results.size()));
    }

    // ── Result types ──────────────────────────────────────────────────────

    public static class ScoredNode {
        public final Node node;
        public final double score;
        public ScoredNode(Node node, double score) {
            this.node  = node;
            this.score = score;
        }
        public double getScore() { return score; }
    }

    public static class ScoredRelation {
        public final String relationName;
        public final double score;
        public ScoredRelation(String relationName, double score) {
            this.relationName = relationName;
            this.score        = score;
        }
        public double getScore() { return score; }
    }
}
