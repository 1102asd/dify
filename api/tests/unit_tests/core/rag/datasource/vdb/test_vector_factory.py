import uuid

from core.rag.datasource.vdb.milvus.milvus_vector import MilvusConfig, MilvusVector
from core.rag.datasource.vdb.qdrant.qdrant_vector import QdrantConfig, QdrantVector
from core.rag.datasource.vdb.weaviate.weaviate_vector import WeaviateConfig, WeaviateVector
from core.rag.models.document import Document
from models.dataset import Dataset


def _get_sample_text() -> str:
    return 'test'


def _get_sample_embeddings() -> list[list[float]]:
    return [[1.1, 2.2, 3.3]]


def _get_sample_query_vector() -> list[float]:
    return _get_sample_embeddings()[0]


def test_weaviate_vector():
    attributes = ['doc_id', 'dataset_id', 'document_id', 'doc_hash']
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
        page_content=_get_sample_text(),
        metadata={
            "doc_id": dataset_id,
            "doc_hash": dataset_id,
            "document_id": dataset_id,
            "dataset_id": dataset_id,
        }
    )

    # create vector
    vector.create(texts=[document], embeddings=_get_sample_embeddings())

    # search by vector
    hits_by_vector = vector.search_by_vector(query_vector=_get_sample_query_vector())
    assert len(hits_by_vector) >= 1

    # search by full text
    hits_by_full_text = vector.search_by_full_text(query=_get_sample_text())
    assert len(hits_by_full_text) >= 1

    # delete vector
    vector.delete()


def test_qdrant_vector():
    dataset_id = str(uuid.uuid4())
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
    # create vector
    vector.create(texts=[document], embeddings=_get_sample_embeddings())

    # search by vector
    hits_by_vector = vector.search_by_vector(query_vector=_get_sample_query_vector())
    assert len(hits_by_vector) >= 1

    # search by full text
    hits_by_full_text = vector.search_by_full_text(query=_get_sample_text())
    assert len(hits_by_full_text) >= 1

    # delete vector
    vector.delete()


def test_milvus_vector():
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

    # create vector
    vector.create(texts=[document], embeddings=_get_sample_embeddings())

    # search by vector
    hits_by_vector = vector.search_by_vector(query_vector=_get_sample_query_vector())
    assert len(hits_by_vector) >= 1

    # milvus dos not support full text searching yet in < 2.3.x

    # delete vector
    vector.delete()
