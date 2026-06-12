"""agorag core library — ETL (build the store) and inference (answer queries).

    from core.etl.etl_pipeline import ETLPipeline
    from core.inference.inference_pipeline import InferencePipeline

Shared building blocks live at the package root: config (paths/constants), ids
(node-id helpers), embeddings (the embedding service) and graph_store (graph
access).
"""
