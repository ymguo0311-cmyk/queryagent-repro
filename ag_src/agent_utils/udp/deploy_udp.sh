#!/bin/bash
# deploy_udp.sh
# 把写好的Java文件复制到neo4j-semantic-udp项目，然后编译
#
# 用法：bash deploy_udp.sh

set -e

PROJECT_DIR="$HOME/neo4j-semantic-udp"
SRC_DIR="$PROJECT_DIR/src/main/java/semantic"

echo "[1/3] Copying Java files to $SRC_DIR ..."
cp SemanticGraphHelper.java "$SRC_DIR/SemanticGraphHelper.java"
cp MatchVertex.java         "$SRC_DIR/MatchVertex.java"
cp MatchEdgePath.java       "$SRC_DIR/MatchEdgePath.java"
echo "      Done."

echo "[2/3] Building with Maven ..."
cd "$PROJECT_DIR"
./mvnw clean package -DskipTests

echo "[3/3] Build complete."
echo ""
echo "Next steps:"
echo "  1. Copy the jar to Neo4j plugins:"
echo "     cp $PROJECT_DIR/target/procedure-template-1.0.0-SNAPSHOT.jar ~/neo4j/plugins/"
echo ""
echo "  2. Restart Neo4j:"
echo "     ~/neo4j/bin/neo4j restart"
echo ""
echo "  3. Test the UDP:"
echo "     cypher-shell -u neo4j -p password \\"
echo "       \"CALL semantic.matchEdgePath([0.1, 0.2, ...], 5) YIELD relationName, score RETURN relationName, score\""
