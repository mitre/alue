# Aerospace Language Understanding Evaluation (ALUE)

<p align="center">
<a href="https://arc.aiaa.org/doi/10.2514/6.2025-3247">
<img src="https://img.shields.io/badge/Read-Paper-green?style=flat" alt="Read Paper" />
</a>
</p>

ALUE (Aerospace Language Understanding Evaluation) is a comprehensive framework designed to facilitate the evaluation and inference of Language Learning Models (LLMs) on aerospace-specific datasets. The framework is user-friendly and versatile, supporting custom datasets, preferred models, user-defined prompts, and quantitative metrics of performance. Contact Eugene Mangortey (emangortey@mitre.org) with inquiries.

## Quickstart

### Installation

This code has only been tested on Python 3.10 and 3.11. Use other versions at your own risk. If you are installing PyTorch, please see [Using PyTorch with uv](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) for more information.

This software uses the [uv](https://docs.astral.sh/uv/) package manager. You can follow the [installation instructions](https://docs.astral.sh/uv/#installation) for your operating system and then run the following to install dependencies:

```sh
$ uv sync
```

Using the command above will automatically create a Python virtual environment in the `.venv` directory.

If you do not wish to use `uv`, then you can install the dependencies using `pip`:

```sh
pip install -r requirements.txt
```
Note that will you need to create your own virtual environment.

### Basic Usage

Once you have installed dependencies, then you need to:

1. Configure your models in `config.py`. See our instructions.
2. Run your models either locally, using [TGI](https://huggingface.co/docs/text-generation-inference/en/), or an OpenAI-compatible endpoint. See our docs below on running models.
3. Run an evaluation script with a dataset of your choice. You can also create your own dataset (see instructions below) .

#### Configuring and Running Models

ALUE supports running models stored locally on disk using available GPUs (full or quantized) or remotely, using the HuggingFace [Text Generation Interface (TGI)](https://huggingface.co/docs/text-generation-inference/en/) or an OpenAI-compatible model endpoint. Users will be able to decide how to run the models when launching any one of the scripts described below.

The full list of models supported should be added to the config.py script.

Example config.py script:

```python
MODELS = {
    "llama_2_7b_chat": "/projects/alue/models/Llama-2-7b-chat-hf",
    "llama_2_13b_chat": "/projects/alue/models/Llama-2-13b-chat-hf",
    "llama_2_70b_chat": {
        "aip_endpoint": "https://llama2-70b.k8s.tld",
        "local_path": ""
        },
    "mistral_v1": "/projects/alue/models/Mistral-7B-Instruct-v0.1",
    "mistral_v2": {
        "aip_endpoint": "https://mistral-7b.k8s.tld",
        "local_path": "/projects/alue/models/Mistral-7B-Instruct-v0.2",
    },
    "mixtral": {
        "aip_endpoint": "https://mixtral-8x7b.k8s.tld",
        "local_path": "/projects/alue/models/Mixtral-8x7B-Instruct-v0.1"
    },
}

EMBEDDING_MODELS = {
    "BAAI/bge-m3": {
        "aip_endpoint": "https://embeddings-bge.k8s.tld/",
        "local_path": ""
    }
}

USE_TGI = True
USE_AIP = False
```

##### Running Models locally

For local models, you can specify the path to the model weights in the `config.py` file. Note that the model weights need to already be available on the local machine.

For example: `"llama_2_7b_chat": "/projects/alue/models/Llama-2-7b-chat-hf"`

For local models, you can specify the path to the model in the `config.py` file.


##### Running Models Remotely

For remote models, you can specify the endpoint in the `config.py` file.

#### Text Generation Inference (TGI):

Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and T5. More info [here](https://huggingface.co/docs/text-generation-inference/en/index).

Using TGI can significantly speed up inference time.

On 586 questions with Mistral-7b-v0.1-Instruct:
Without TGI: 00:15:45

With TGI: 00:04:43

#### Setting up TGI

To use the TGI endpoint, first modify the `run_tgi.sh` file to configure it to your needs in terms of GPU usage. Then run the script with the name of the model as an argument. The name of the model is the model directory name in `/path/to/models` (although this can be changed with a minor modification). For example, to run the TGI endpoint with Mistral V1, in `/projects/alue/models/Mistral-7B-Instruct-v0.1`, run the script as `sudo ./run_tgi.sh Mistral-7B-Instruct-v0.1`.


**NOTE***: When running models quantized, add `quantize bitsandbytes-nf4` at the end of the `docker run command`. It wil look like the following:
```bash
docker run -d \
    --name tgi \
    --gpus all \
    --shm-size 1g \
    -p 3000:80 -v $volume:/data \
    --rm <docker_image_url>:$tgi_version \
    --model-id $model_id \
    --huggingface-hub-cache /data \
    --num-shard $num_shard \
    --max-input-length $max_input_length \
    --max-batch-prefill-tokens $max_batch_prefill_tokens \
    --max-total-tokens $max_total_tokens \
    --quantize bitsandbytes-nf4
```

Run the script once you have modified it with the following command:
```bash
sudo ./run_tgi.sh
```

Make sure to replace any paths with those for your setup.

#### Running Models on Endpoints

The current ALUE configuration allows users to run models on endpoints.

Please note the endpoints have to be hosted via TGI. This is a bit different than running the TGI local instance described above. The endpoints are accessible via the internet instead of running on localhost as with TGI local. The endpoints are listed in the `config.py` script.

If you have endpoints set up, ensure that the model has an associated `aip_endpoint` in the `config.py` script.

Example: `"llama_2_7b_chat": "https://llama-2-7b-chat.k8s.tld"`.

## Contributing

ALUE is an open-science initiative and community contributions will help us further ALUE and LLM usage on aerospace data. We welcome:

    🔧 New Tools: Specialized analysis functions and algorithms
    📊 Datasets: Curated aerospace data and knowledge bases
    💻 Software: Integration of existing LLM evaluation software packages
    📋 Benchmarks: Evaluation datasets and performance metrics
    📚 Misc: Tutorials, examples, and use cases
    🔧 Update existing tools: many current tools, benchmarks, metrics are not optimized - fixes and replacements are welcome!

Check out the [Contributing Guide](CONTRIBUTING.md) on how to contribute to the ALUE project.

If you have particular tool/database/software in mind that you want to add, you can also submit to this form and the ALUE team may implement them depending on our resource constraints.



## Release Schedule

To Be Determined.

## Note

- This release was frozen as of July 21, 2025 and was intended for the AIAA AVIATION 2025 conference.
- ALUE itself is Apache 2.0-licensed, but certain integrated tools, databases, or software may carry more restrictive commercial licenses. Review each component carefully before any commercial use.
- The MITRE Corporation and the Federal Aviation Administration (FAA) make no claims that this software will be maintained. Continued development is pursuant to FAA priorities, needs, and available funding.

## Citation

You can cite this work using either:

Eugene Mangortey, Satyen Singh, Shuo Chen and Kunal Sarkhel. "Aviation Language Understanding Evaluation (ALUE) – Large Language Model Benchmark with Aviation Datasets," AIAA 2025-3247. _AIAA AVIATION FORUM AND ASCEND 2025._ July 2025.

It can also be cited using BibTeX.

```
@inbook{doi:10.2514/6.2025-3247,
author = {Eugene Mangortey and Satyen Singh and Shuo Chen and Kunal Sarkhel},
title = {Aviation Language Understanding Evaluation (ALUE) – Large Language Model Benchmark with Aviation Datasets},
booktitle = {AIAA AVIATION FORUM AND ASCEND 2025},
chapter = {},
pages = {},
doi = {10.2514/6.2025-3247},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2025-3247},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2025-3247},
    abstract = { Large Language Models (LLMs) present revolutionary potential for the aviation industry, enabling stakeholders to derive critical intelligence and improve operational efficiencies through automation. However, given the safety-critical nature of aviation, a rigorous domain-specific evaluation of LLMs is paramount before their integration into workflows. General-purpose LLM benchmarks often do not capture the nuanced understanding of aerospace-specific knowledge and the phraseology required for reliable application. This paper introduces the Aerospace Language Understanding Evaluation (ALUE) benchmark, an aviation-specific framework designed for scalable evaluation, assessment, and benchmarking of LLMs against specialized aviation datasets and language tasks. ALUE incorporates diverse datasets and tasks, including binary and multiclass classification for hazard identification, extractive question answering for precise information retrieval (e.g., tail numbers, runways), sentiment analysis, and multiclass token classification for fine-grained analysis of air traffic control communications. ALUE also introduces several metrics for evaluating the correctness of generated responses utilizing LLMs to identify and judge claims made in generated responses. Our findings demonstrate that structured prompts and in-context examples significantly improve model performance, highlighting that general models struggle with aviation tasks without such guidance and often produce verbose or unstructured outputs. ALUE provides a crucial tool for guiding the development and safe deployment of LLMs tailored to the unique demands of the aviation and aerospace domains. }
}
```

If you'd like to cite this work using a citation manager such as [Zotero](https://www.zotero.org/), please see the [CITATION.cff](CITATION.cff) file for more information citing this work. For more informaton on the CITATION.cff file format, please see [What is a `CITATION.cff` file?](https://citation-file-format.github.io/)

# Copyright Notices

This is the copyright work of The MITRE Corporation, and was produced for the U. S. Government under Contract Number 693KA8-22-C-00001, and is subject to Federal Aviation Administration Acquisition Management System Clause 3.5-13, Rights In Data-General (Oct. 2014), Alt. III and Alt. IV (Jan. 2009). No other use other than that granted to the U. S. Government, or to those acting on behalf of the U. S. Government, under that Clause is authorized without the express written permission of The MITRE Corporation. For further information, please contact The MITRE Corporation, Contracts Management Office, 7515 Colshire Drive, McLean, VA 22102-7539, (703) 983-6000.

&copy; 2025 The MITRE Corporation. All Rights Reserved.

Approved for Public Release, Distribution Unlimited. PRS Case 25-2078.
