"""Inference: answer a query over the persistent store.

    Retriever   -> embed the query + fetch top-k findings from Qdrant
    Linker      -> stream a gpt-5.4 acknowledgment + link query to known entities
    GraphParser -> stream a gpt-5.5 answer + emit the validated knowledge graph
    core_llm    -> the shared streaming LLM client and the prompts both calls use

InferencePipeline composes them into the event stream the backend consumes.
"""
