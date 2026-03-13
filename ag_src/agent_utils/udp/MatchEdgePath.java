package semantic;

import java.util.List;
import java.util.stream.Stream;

import org.neo4j.graphdb.Transaction;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

/**
 * UDP: semantic.matchEdgePath
 *
 * Finds the top-K most semantically similar Freebase relations
 * given a query embedding vector.
 *
 * THIS IS THE CORE INNOVATION:
 * Original QueryAgent uses BM25 keyword matching (faiss_filter) to rank relations.
 * This UDP replaces it with vector similarity search over 24K+ relation embeddings.
 * This solves the "Schema Drift" / "Relation Mismatch" problem:
 *   - BM25: "railway terminus" → only finds exact keyword matches
 *   - UDP:  "railway terminus" → finds semantically related relations
 *             e.g. transport.transit_line.terminus  (exact)
 *                  transport.transit_stop.stop_type (semantic)
 *                  rail.railway_line.stations       (semantic)
 *
 * Usage in Cypher:
 *   CALL semantic.matchEdgePath($queryEmbedding, 10)
 *   YIELD relationName, score
 *   RETURN relationName, score
 *   ORDER BY score DESC
 *
 * Node schema required in Neo4j:
 *   (:Relation {name: "transport.transit_line.terminus", embedding: [...]})
 *   → Created by ingest_relations.py using fb_relation.txt + sentence-transformer
 */
public class MatchEdgePath {

    @Context
    public Transaction tx;

    @Context
    public Log log;

    // ── Procedure ─────────────────────────────────────────────────────────
    @Procedure(name = "semantic.matchEdgePath", mode = Mode.READ)
    @Description(
        "Find top-K Freebase relations semantically similar to the query embedding. " +
        "Replaces BM25 keyword matching with vector similarity search. " +
        "Solves the Schema Drift problem in KBQA."
    )
    public Stream<MatchEdgePathResult> matchEdgePath(
            @Name("queryEmbedding") List<Double> queryEmbedding,
            @Name(value = "topK", defaultValue = "10") Long topK) {

        log.debug("semantic.matchEdgePath called, topK=" + topK);

        if (queryEmbedding == null || queryEmbedding.isEmpty()) {
            log.warn("semantic.matchEdgePath: queryEmbedding is null or empty");
            return Stream.empty();
        }

        List<SemanticGraphHelper.ScoredRelation> results =
            SemanticGraphHelper.matchRelation(tx, queryEmbedding, topK);

        return results.stream()
            .map(sr -> new MatchEdgePathResult(sr.relationName, sr.score));
    }

    // ── Result record ─────────────────────────────────────────────────────
    public static class MatchEdgePathResult {
        /** Freebase relation name e.g. "transport.transit_line.terminus" */
        public String relationName;
        /** Cosine similarity score [0, 1] */
        public double score;

        public MatchEdgePathResult(String relationName, double score) {
            this.relationName = relationName;
            this.score        = score;
        }
    }
}
