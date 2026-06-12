"""ETL: turn Agora's publications into the persistent store.

    Reader      -> read source publications (scrape or sample) + diff vs. stored
    Transformer -> build the knowledge graph + embeddable chunks
    Loader      -> persist publications/graph/entities + embed chunks into Qdrant

ETLPipeline wires the three together; run `python -m core.etl.etl_pipeline`.
"""
