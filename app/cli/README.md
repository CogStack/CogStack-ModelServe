# `python cli.py stream`

This groups various stream operations

**Usage**:

```console
$ python cli.py stream [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `chat`: This gets NER entities by chatting with...
* `json_lines`: This gets NER entities as a JSON Lines stream

## `python cli.py stream chat`

This gets NER entities by chatting with the model

**Usage**:

```console
$ python cli.py stream chat [OPTIONS]
```

**Options**:

* `--base-url TEXT`: The CMS base url  [default: ws://127.0.0.1:8000]
* `--help`: Show this message and exit.

## `python cli.py stream json_lines`

This gets NER entities as a JSON Lines stream

**Usage**:

```console
$ python cli.py stream json_lines [OPTIONS]
```

**Options**:

* `--jsonl-file-path TEXT`: The path to the JSON Lines file  [required]
* `--base-url TEXT`: The CMS base url  [default: http://127.0.0.1:8000]
* `--timeout-in-secs INTEGER`: The max time to wait before disconnection  [default: 0]
* `--help`: Show this message and exit.
