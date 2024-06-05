# `python cli.py`

CLI for various CogStack ModelServe operations

**Usage**:

```console
$ python cli.py [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `export-model-apis`: This generates model-specific API docs for...
* `export-openapi-spec`: This generates a single API doc for all...
* `register`: This pushes a pretrained NLP model to the...
* `serve`: This serves various CogStack NLP models

## `python cli.py export-model-apis`

This generates model-specific API docs for enabled endpoints

**Usage**:

```console
$ python cli.py export-model-apis [OPTIONS]
```

**Options**:

* `--model-type [medcat_snomed|medcat_umls|medcat_icd10|medcat_deid|transformers_deid]`: The type of the model to serve  [required]
* `--add-training-apis / --no-add-training-apis`: Add training APIs to the doc  [default: no-add-training-apis]
* `--add-evaluation-apis / --no-add-evaluation-apis`: Add evaluation APIs to the doc  [default: no-add-evaluation-apis]
* `--add-previews-apis / --no-add-previews-apis`: Add preview APIs to the doc  [default: no-add-previews-apis]
* `--add-user-authentication / --no-add-user-authentication`: Add user authentication APIs to the doc  [default: no-add-user-authentication]
* `--exclude-unsupervised-training / --no-exclude-unsupervised-training`: Exclude the unsupervised training API  [default: no-exclude-unsupervised-training]
* `--exclude-metacat-training / --no-exclude-metacat-training`: Exclude the metacat training API  [default: no-exclude-metacat-training]
* `--model-name TEXT`: The string representation of the model name
* `--help`: Show this message and exit.

## `python cli.py export-openapi-spec`

This generates a single API doc for all endpoints

**Usage**:

```console
$ python cli.py export-openapi-spec [OPTIONS]
```

**Options**:

* `--api-title TEXT`: The string representation of the API title  [default: CogStack Model Serve APIs]
* `--help`: Show this message and exit.

## `python cli.py register`

This pushes a pretrained NLP model to the CogStack ModelServe registry

**Usage**:

```console
$ python cli.py register [OPTIONS]
```

**Options**:

* `--model-type [medcat_snomed|medcat_umls|medcat_icd10|medcat_deid|transformers_deid]`: The type of the model to serve  [required]
* `--model-path TEXT`: The file path to the model package  [required]
* `--model-name TEXT`: The string representation of the registered model  [required]
* `--training-type TEXT`: The type of training the model went through
* `--model-config TEXT`: The string representation of a JSON object
* `--model-metrics TEXT`: The string representation of a JSON array
* `--model-tags TEXT`: The string representation of a JSON object
* `--help`: Show this message and exit.

## `python cli.py serve`

This serves various CogStack NLP models

**Usage**:

```console
$ python cli.py serve [OPTIONS]
```

**Options**:

* `--model-type [medcat_snomed|medcat_umls|medcat_icd10|medcat_deid|transformers_deid]`: The type of the model to serve  [required]
* `--model-path TEXT`: The file path to the model package
* `--mlflow-model-uri models:/MODEL_NAME/ENV`: The URI of the MLflow model to serve
* `--host TEXT`: The hostname of the server  [default: 127.0.0.1]
* `--port TEXT`: The port of the server  [default: 8000]
* `--model-name TEXT`: The string representation of the model name
* `--help`: Show this message and exit.
