import uuid
from typing import Optional

from core.rag.datasource.vdb.milvus.milvus_vector import MilvusConfig, MilvusVector
from core.rag.datasource.vdb.qdrant.qdrant_vector import QdrantConfig, QdrantVector
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.weaviate.weaviate_vector import WeaviateConfig, WeaviateVector
from core.rag.models.document import Document
from models.dataset import Dataset


def test_weviate_vector():
    vector_type = 'weaviate'
    attributes = ['doc_id', 'dataset_id', 'document_id', 'doc_hash']

    vector: Optional[BaseVector] = None
    if vector_type == "weaviate":
        dataset_id = str(uuid.uuid4())
        collection_name = Dataset.gen_collection_name_by_id(dataset_id)
        vector = WeaviateVector(
            collection_name=collection_name,
            config=WeaviateConfig(
                endpoint='http://localhost:8080',
                api_key='WVF5YThaHlkYwhGUSmCRgsX3tD5ngdN8pkih',
                batch_size=100
            ),
            attributes=attributes
        )
        document = Document(
            page_content='test',
            metadata={
                "doc_id": dataset_id,
                "doc_hash": dataset_id,
                "document_id": dataset_id,
                "dataset_id": dataset_id,
            }
        )
        vector.create(texts=[document], embeddings=[[1.1, 2.2, 3.3]])
        vector.delete()


def test_qdrant_vector():
    vector_type = 'qdrant'

    vector: Optional[BaseVector] = None
    if vector_type == "qdrant":
        dataset_id = str(uuid.uuid4())
        collection_name = Dataset.gen_collection_name_by_id(dataset_id)
        vector = QdrantVector(
            collection_name=dataset_id,
            group_id=dataset_id,
            config=QdrantConfig(
                endpoint='http://localhost:6333',
                api_key='difyai123456',
                root_path='/',
                timeout='20'
            )
        )
        document = Document(
            page_content='test',
            metadata={
                "doc_id": dataset_id,
                "doc_hash": dataset_id,
                "document_id": dataset_id,
                "dataset_id": dataset_id,
            }
        )
        vector.create(texts=[document], embeddings=[[1.1, 2.2, 3.3]])
        vector.delete()


def test_milvus_vector():
    vector_type = 'qdrant'

    vector: Optional[BaseVector] = None
    if vector_type == "qdrant":
        dataset_id = str(uuid.uuid4())
        collection_name = Dataset.gen_collection_name_by_id(dataset_id)
        vector = MilvusVector(
            collection_name=collection_name,
            config=MilvusConfig(
                host='localhost',
                port='19530',
                user='root',
                password='Milvus',
                secure='false',
                database='default',
            )
        )
        document = Document(
            page_content='test',
            metadata={
                "doc_id": dataset_id,
                "doc_hash": dataset_id,
                "document_id": dataset_id,
                "dataset_id": dataset_id,
            }
        )
        vector.create(texts=[document], embeddings=[[1.1, 2.2, 3.3]])
        vector.delete()
