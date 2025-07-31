from prometheus_client import Histogram, Gauge

# The histogram metric to track the number of annotations extracted from a document by different handlers
cms_doc_annotations = Histogram(
    "cms_doc_annotations",
    "The number of annotations extracted from a document",
    ["handler"],
)

# The gauge metric to track the average accuracy of annotations extracted from a document by different handlers
cms_avg_anno_acc_per_doc = Gauge(
    "cms_avg_anno_acc_per_doc",
    "The average accuracy of annotations extracted from a document",
    ["handler"],
)

# The gauge metric to track the average accuracy of annotations for a specific concept by different handlers
cms_avg_anno_acc_per_concept = Gauge(
    "cms_avg_anno_acc_per_concept",
    "The average accuracy of annotations for a specific concept",
    ["handler", "concept"],
)

# The gauge metric to track the average confidence of meta annotations extracted from a document by different handlers
cms_avg_meta_anno_conf_per_doc = Gauge(
    "cms_avg_meta_anno_conf_per_doc",
    "The average confidence of meta annotations extracted from a document",
    ["handler"],
)

# The histogram metric to track the number of bulk-processed documents by different handlers
cms_bulk_processed_docs = Histogram(
    "cms_bulk_processed_docs",
    "Number of bulk-processed documents",
    ["handler"],
)

# The histogram metric to track the number of tokens in the messages of the input prompt
cms_prompt_tokens = Histogram(
    "cms_prompt_tokens",
    "Number of tokens in the messages of the input prompt",
    ["handler"],
)

# The histogram metric to track the number of tokens in the generated assistant reply
cms_completion_tokens = Histogram(
    "cms_completion_tokens",
    "Number of tokens in the generated assistant reply",
    ["handler"],
)

# The histogram metric to track the total number of tokens used in the prompt and the completion
cms_total_tokens = Histogram(
    "cms_total_tokens",
    "Number of tokens used in the prompt and the completion",
    ["handler"],
)
