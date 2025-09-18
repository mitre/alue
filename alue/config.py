# noqa: D100, E501

# Change these model endpoints to the ones relevant to your project  # noqa: D100

MODELS = {
    "model_name": {
        "aip_endpoint": "https://endpoint-url.tld/v1",
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "aip_endpoint": "https://endpoint-url.tld/v1"
    },
    "llama_2_70b_chat": {
        "aip_endpoint": "https://endpoint-url.tld/v1",
        "local_path": "",
    },
    "llama_3_8b": "/projects/aviationllm/llm/models/Llama-3.1-8B",
    "llama_3_8b_instruct": {
        "aip_endpoint": "https://llama3-8b.k8s.tld",
        "local_path": "/home/alue/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    },
}

EMBEDDING_MODELS = {
    "BAAI/bge-m3": {
        "aip_endpoint": "https://embeddings-url.tld/",
        "local_path": "/path/to/embedding/model",
    }
}

USE_TGI = True
USE_AIP = False
