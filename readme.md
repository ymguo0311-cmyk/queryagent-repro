# QueryAgent + Neo4j Semantic Query System

An extension of the QueryAgent framework integrated with Neo4j for semantic query processing over graph databases. This project combines LLM-powered natural language understanding with custom Neo4j user-defined procedures (UDP) to enable intuitive querying of knowledge graphs.

## Project Overview

QueryAgent is a system that translates natural language queries into database-specific query languages using Large Language Models. This repository extends the original QueryAgent implementation to support Neo4j graph databases with custom semantic matching capabilities.

## What I Built

### Core Components

1. **Neo4j Semantic UDP (User-Defined Procedure)**
   - Custom Java procedures for semantic path matching in Neo4j
   - Enables multi-hop reasoning over knowledge graph relationships
   - Optimized for semantic similarity queries

2. **QueryAgent Integration**
   - Modified query translation pipeline to generate Cypher queries
   - Adapted prompting strategies for graph database queries
   - Integrated semantic matching with LLM-generated queries

### Technical Implementation

- Extended QueryAgent's query parsing to handle graph traversal patterns
- Designed semantic matching procedures for Neo4j (implementation details in research repository)
- Created integration layer connecting LLM query generation with graph database execution

## Tech Stack

**Core Framework**: QueryAgent (original implementation)  
**Database**: Neo4j (with custom UDP extensions)  
**Language**: Java (UDP), Python (QueryAgent)  
**LLM Integration**: OpenAI API / Compatible LLM endpoints  

## Project Status

**Current**: Proof-of-concept implementation demonstrating semantic query capabilities over Neo4j knowledge graphs.

**In Progress**: 
- Performance optimization for large-scale graphs
- Extended semantic operators for complex reasoning patterns

## Use Case Example

**Natural Language Query**: "Find research papers related to semantic search"

**System Flow**:
1. LLM translates to Cypher query with semantic matching
2. Neo4j UDP executes semantic path traversal
3. Results ranked by semantic relevance

## Background

This work builds upon research in hybrid database systems for AI applications, exploring how LLMs can enhance structured database querying through semantic understanding.

## References

- Original QueryAgent: [https://github.com/cdhx/QueryAgent]
- Related research: Hybrid database systems, semantic query processing, knowledge graphs

## Note

This repository demonstrates the integration architecture and workflow. The Neo4j semantic matching procedures were developed as part of ongoing research at HKUST and are not included in this public repository. The focus here is on the QueryAgent extension and integration strategy.

## Attribution
   
   - **QueryAgent Core**: Based on the original QueryAgent implementation
   - **Neo4j Semantic UDP**: Custom implementation developed as part of UROP research under Prof. Zhou Xiaofang at HKUST
   - **Integration Layer**: Original work combining both systems
