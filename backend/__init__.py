"""Backend — a thin layer over the core library for the gradio app.

Most logic lives in `core` (ETL + inference). This package only loads the
persistent store, wires up the inference pipeline, and exposes the few extras the
gradio UI needs (publication lookups, citation URLs). See backend.service.AppService.
"""
