from prometheus_client import Histogram, Gauge, Counter

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

# The histogram metric to track Time To First Token (TTFT) in milliseconds
cms_ttft_milliseconds = Histogram(
    "cms_ttft_milliseconds",
    "Time to first generated token in milliseconds",
    ["handler"],
)

# The histogram metric to track Time Per Output Token (TPOT) in milliseconds
cms_tpot_milliseconds = Histogram(
    "cms_tpot_milliseconds",
    "Average time per output token in milliseconds",
    ["handler"],
)

# The histogram metric to track end-to-end generation request latency in milliseconds
cms_gen_request_latency_milliseconds = Histogram(
    "cms_gen_request_latency_milliseconds",
    "End-to-End generation request latency in milliseconds",
    ["handler"],
)

# The counter metric to track the number of prefix cache queries
cms_prefix_cache_queries = Counter(
    "cms_prefix_cache_queries",
    "Total number of prefix cache queries",
    ["handler"],
)

# The counter metric to track the number of prefix cache hits
cms_prefix_cache_hits = Counter(
    "cms_prefix_cache_hits",
    "Total number of prefix cache hits",
    ["handler"],
)

# The histogram metric to track generation job queue time in milliseconds
cms_gen_job_queue_time_milliseconds = Histogram(
    "cms_gen_job_queue_time_milliseconds",
    "Time spent waiting in the queue before generation starts in milliseconds",
    ["handler"],
)

# The gauge metric to track the number of generation jobs actively running
cms_num_of_gen_jobs_running = Gauge(
    "cms_num_of_gen_jobs_running",
    "Number of generation jobs currently running",
    ["handler"],
)

# The gauge metric to track the number of generation jobs waiting in the queue
cms_num_of_gen_jobs_waiting = Gauge(
    "cms_num_of_gen_jobs_waiting",
    "Number of generation jobs currently waiting in the queue",
    ["handler"],
)
