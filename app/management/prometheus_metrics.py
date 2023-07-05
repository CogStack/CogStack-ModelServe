from prometheus_client import Histogram, Gauge

cms_doc_annotations = Histogram("cms_doc_annotations", "Number of annotations extracted from a document", ["handler"])
cms_avg_anno_acc_per_doc = Gauge("cms_avg_anno_acc_per_doc", "The average accuracy of annotations extracted from a document", ["handler"])
cms_avg_meta_anno_conf_per_doc = Gauge("cms_avg_meta_anno_conf_per_doc", "The average confidence of meta annotations extracted from a document", ["handler"])
cms_bulk_processed_docs = Histogram("cms_bulk_processed_docs", "Number of bulk-processed documents", ["handler"])
