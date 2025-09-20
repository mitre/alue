# noqa: D100, E501

# Change these model endpoints to the ones relevant to your project  # noqa: D100

from settings import settings

MODELS = {
    "model_name": {
        "endpoint": "https://endpoint-url.tld/v1",
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "endpoint": "https://endpoint-url.tld/v1"
    },
    "llama_2_70b_chat": {
        "endpoint": "https://endpoint-url.tld/v1",
        "local_path": "",
    },
    "llama_3_8b": "/projects/aviationllm/llm/models/Llama-3.1-8B",
    "llama_3_8b_instruct": {
        "endpoint": "https://llama3-8b.k8s.tld",
        "local_path": "",
    },
}

EMBEDDING_MODELS = {
    "BAAI/bge-m3": {
        "endpoint": {settings.EMBEDDINGS_URL or "http://embeddings-url.tld/"},
        "local_path": "/path/to/embedding/model",
    }
}

ENDPOINT_TYPE = settings.ENDPOINT_TYPE
ENDPOINT_URL = settings.ENDPOINT_URL


