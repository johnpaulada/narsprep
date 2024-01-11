import logging

from app.engine.constants import DATA_DIR, STORAGE_DIR
from app.engine.context import create_service_context
from dotenv import load_dotenv

load_dotenv()

from llama_index import SimpleDirectoryReader, VectorStoreIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_datasource(service_context):
    logger.info("Creating new index")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, show_progress=True
    )
    index.storage_context.persist(STORAGE_DIR)
    logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")


if __name__ == "__main__":
    service_context = create_service_context()
    generate_datasource(service_context)
