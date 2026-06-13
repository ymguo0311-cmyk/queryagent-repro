package semantic;

import java.util.List;
import java.util.stream.Stream;

import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

/**
 * UDP: semantic.matchVertex
 *
 * Finds the top-K most semantically similar entity nodes
 * given a query embedding vector.
 *
 * Usage in Cypher:
 *   CALL semantic.matchVertex($queryEmbedding, 5)
 *   YIELD vertex, score
 *   RETURN vertex.mid, vertex.name, score
 *
 * This replaces exact entity lookup with vector similarity search,
 * solving the entity surface form mismatch problem.
 */
public class MatchVertex {

    @Context
    public Transaction tx;

    @Context
    public Log log;

    // ── Procedure ─────────────────────────────────────────────────────────
    @Procedure(name = "semantic.matchVertex", mode = Mode.READ)
    @Description(
        "Find top-K entity nodes semantically similar to the query embedding. " +
        "Uses Neo4j vector index if available, falls back to cosine similarity scan."
    )
    public Stream<MatchVertexResult> matchVertex(
            @Name("queryEmbedding") List<Double> queryEmbedding,
            @Name(value = "topK", defaultValue = "5") Long topK) {

        log.debug("semantic.matchVertex called, topK=" + topK);

        if (queryEmbedding == null || queryEmbedding.isEmpty()) {
            log.warn("semantic.matchVertex: queryEmbedding is null or empty");
            return Stream.empty();
        }

        List<SemanticGraphHelper.ScoredNode> results =
            SemanticGraphHelper.matchVertex(tx, queryEmbedding, topK);

        return results.stream().map(sn -> new MatchVertexResult(sn.node, sn.score));
    }

    // ── Result record ─────────────────────────────────────────────────────
    public static class MatchVertexResult {
        /** The matched entity node */
        public Node vertex;
        /** Cosine similarity score [0, 1] */
        public double score;

        public MatchVertexResult(Node vertex, double score) {
            this.vertex = vertex;
            this.score  = score;
        }
    }
}
