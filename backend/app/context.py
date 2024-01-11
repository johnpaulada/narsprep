import os

from llama_index import ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama, OpenAI


def create_base_context():
    model = os.getenv("MODEL", "gpt-3.5-turbo")
    return ServiceContext.from_defaults(
        llm=Ollama(model="meditron", temperature=0),
        embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5"),
    )
