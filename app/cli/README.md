# `cms`

CLI for various CogStack ModelServe operations

**Usage**:

```console
$ cms [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `serve`: This serves various CogStack NLP models
* `train`: This pretrains or fine-tunes various...
* `register`: This pushes a pretrained NLP model to the...
* `export-model-apis`: This generates a model-specific API...
* `build`: This builds an OCI-compliant image to...
* `export-openapi-spec`: This generates an API document for all...
* `stream`: This groups various stream operations
* `package`: This groups various package operations

## `cms serve`

This serves various CogStack NLP models

**Usage**:

```console
$ cms serve [OPTIONS]
```

**Options**:

* `--model-type [medcat_snomed|medcat_umls|medcat_icd10|medcat_opcs4|medcat_deid|anoncat|transformers_deid|huggingface_ner]`: The type of the model to serve  [required]
* `--model-path TEXT`: The file path to the model package
* `--mlflow-model-uri models:/MODEL_NAME/ENV`: The URI of the MLflow model to serve
* `--host TEXT`: The hostname of the server  [default: 127.0.0.1]
* `--port TEXT`: The port of the server  [default: 8000]
* `--model-name TEXT`: The string representation of the model name
* `--streamable / --no-streamable`: Serve the streamable endpoints only  [default: no-streamable]
* `--device [default|cpu|cuda|mps]`: The device to serve the model on  [default: default]
* `--debug / --no-debug`: Run in the debug mode
* `--help`: Show this message and exit.

## `cms train`

This pretrains or fine-tunes various CogStack NLP models

**Usage**:

```console
$ cms train [OPTIONS]
```

**Options**:

* `--model-type [medcat_snomed|medcat_umls|medcat_icd10|medcat_opcs4|medcat_deid|anoncat|transformers_deid|huggingface_ner]`: The type of the model to train  [required]
* `--base-model-path TEXT`: The file path to the base model package to be trained on
* `--mlflow-model-uri models:/MODEL_NAME/ENV`: The URI of the MLflow model to train
* `--training-type [supervised|unsupervised|meta_supervised]`: The type of training  [required]
* `--data-file-path TEXT`: The path to the training asset file  [required]
* `--epochs INTEGER`: The number of training epochs  [default: 1]
* `--log-frequency INTEGER`: The number of processed documents or epochs after which training metrics will be logged  [default: 1]
* `--hyperparameters TEXT`: The overriding hyperparameters serialised as JSON string  [default: {}]
* `--description TEXT`: The description of the training or change logs
* `--model-name TEXT`: The string representation of the model name
* `--device [default|cpu|cuda|mps]`: The device to train the model on  [default: default]
* `--debug / --no-debug`: Run in the debug mode
* `--help`: Show this message and exit.

## `cms register`

This pushes a pretrained NLP model to the CogStack ModelServe registry

**Usage**:

```console
$ cms register [OPTIONS]
```

**Options**:

* `--model-type [medcat_snomed|medcat_umls|medcat_icd10|medcat_opcs4|medcat_deid|anoncat|transformers_deid|huggingface_ner]`: The type of the model to register  [required]
* `--model-path TEXT`: The file path to the model package  [required]
* `--model-name TEXT`: The string representation of the registered model  [required]
* `--training-type [supervised|unsupervised|meta_supervised]`: The type of training the model went through
* `--model-config TEXT`: The string representation of a JSON object
* `--model-metrics TEXT`: The string representation of a JSON array
* `--model-tags TEXT`: The string representation of a JSON object
* `--debug / --no-debug`: Run in the debug mode
* `--help`: Show this message and exit.

## `cms export-model-apis`

This generates a model-specific API document for enabled endpoints

**Usage**:

```console
$ cms export-model-apis [OPTIONS]
```

**Options**:

* `--model-type [medcat_snomed|medcat_umls|medcat_icd10|medcat_opcs4|medcat_deid|anoncat|transformers_deid|huggingface_ner]`: The type of the model to serve  [required]
* `--add-training-apis / --no-add-training-apis`: Add training APIs to the doc  [default: no-add-training-apis]
* `--add-evaluation-apis / --no-add-evaluation-apis`: Add evaluation APIs to the doc  [default: no-add-evaluation-apis]
* `--add-previews-apis / --no-add-previews-apis`: Add preview APIs to the doc  [default: no-add-previews-apis]
* `--add-user-authentication / --no-add-user-authentication`: Add user authentication APIs to the doc  [default: no-add-user-authentication]
* `--exclude-unsupervised-training / --no-exclude-unsupervised-training`: Exclude the unsupervised training API  [default: no-exclude-unsupervised-training]
* `--exclude-metacat-training / --no-exclude-metacat-training`: Exclude the metacat training API  [default: no-exclude-metacat-training]
* `--model-name TEXT`: The string representation of the model name
* `--help`: Show this message and exit.

## `cms build`

This builds an OCI-compliant image to containerise CMS

**Usage**:

```console
$ cms build [OPTIONS]
```

**Options**:

* `--dockerfile-path TEXT`: The path to the Dockerfile  [required]
* `--context-dir TEXT`: The directory containing the set of files accessible to the build  [required]
* `--model-name TEXT`: The string representation of the model name  [default: CMS model]
* `--user-id INTEGER`: The ID for the non-root user  [default: 1000]
* `--group-id INTEGER`: The group ID for the non-root user  [default: 1000]
* `--http-proxy TEXT`: The string representation of the HTTP proxy
* `--https-proxy TEXT`: The string representation of the HTTPS proxy
* `--no-proxy TEXT`: The string representation of addresses by-passing proxies  [default: localhost,127.0.0.1]
* `--version-tag TEXT`: The version tag of the built image  [default: latest]
* `--backend [docker build|docker buildx build]`: The backend used for building the image  [default: docker build]
* `--help`: Show this message and exit.

## `cms export-openapi-spec`

This generates an API document for all endpoints defined in CMS

**Usage**:

```console
$ cms export-openapi-spec [OPTIONS]
```

**Options**:

* `--api-title TEXT`: The string representation of the API title  [default: CogStack Model Serve APIs]
* `--help`: Show this message and exit.

## `cms stream`

This groups various stream operations

**Usage**:

```console
$ cms stream [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `json-lines`: This gets NER entities as a JSON Lines stream
* `chat`: This gets NER entities by chatting with...

### `cms stream json-lines`

This gets NER entities as a JSON Lines stream

**Usage**:

```console
$ cms stream json-lines [OPTIONS]
```

**Options**:

* `--jsonl-file-path TEXT`: The path to the JSON Lines file  [required]
* `--base-url TEXT`: The CMS base url  [default: http://127.0.0.1:8000]
* `--timeout-in-secs INTEGER`: The max time to wait before disconnection  [default: 0]
* `--debug / --no-debug`: Run in the debug mode
* `--help`: Show this message and exit.

### `cms stream chat`

This gets NER entities by chatting with the model

**Usage**:

```console
$ cms stream chat [OPTIONS]
```

**Options**:

* `--base-url TEXT`: The CMS base url  [default: ws://127.0.0.1:8000]
* `--debug / --no-debug`: Run in the debug mode
* `--help`: Show this message and exit.

## `cms package`

This groups various package operations

**Usage**:

```console
$ cms package [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `hf-model`: This packages a remotely hosted or locally...
* `hf-dataset`: This packages a remotely hosted or locally...

### `cms package hf-model`

This packages a remotely hosted or locally cached Hugging Face model into a model package

**Usage**:

```console
$ cms package hf-model [OPTIONS]
```

**Options**:

* `--hf-repo-id TEXT`: The repository ID of the model to download from Hugging Face Hub, e.g., &#x27;google-bert/bert-base-cased&#x27;
* `--hf-repo-revision TEXT`: The revision of the model to download from Hugging Face Hub
* `--cached-model-dir TEXT`: The path to the cached model directory, will only be used if --hf-repo-id is not provided
* `--output-model-package TEXT`: The path where the model package will be saved, minus any format-specific extension, e.g., &#x27;./model_packages/bert-base-cased&#x27;
* `--archive-format [zip|gztar]`: The archive format of the model package, e.g., &#x27;zip&#x27; or &#x27;gztar&#x27;  [default: zip]
* `--remove-cached / --no-remove-cached`: Whether to remove the downloaded cache after the model package is saved  [default: no-remove-cached]
* `--help`: Show this message and exit.

### `cms package hf-dataset`

This packages a remotely hosted or locally cached Hugging Face dataset into a dataset package

**Usage**:

```console
$ cms package hf-dataset [OPTIONS]
```

**Options**:

* `--hf-dataset-id TEXT`: The repository ID of the dataset to download from Hugging Face Hub, e.g., &#x27;stanfordnlp/imdb&#x27;
* `--hf-dataset-revision TEXT`: The revision of the dataset to download from Hugging Face Hub
* `--cached-dataset-dir TEXT`: The path to the cached dataset directory, will only be used if --hf-dataset-id is not provided
* `--output-dataset-package TEXT`: The path where the dataset package will be saved, minus any format-specific extension, e.g., &#x27;./dataset_packages/imdb&#x27;
* `--archive-format [zip|gztar]`: The archive format of the dataset package, e.g., &#x27;zip&#x27; or &#x27;gztar&#x27;  [default: zip]
* `--remove-cached / --no-remove-cached`: Whether to remove the downloaded cache after the dataset package is saved  [default: no-remove-cached]
* `--trust-remote-code / --no-trust-remote-code`: Whether to trust and use the remote script of the dataset  [default: no-trust-remote-code]
* `--help`: Show this message and exit.
