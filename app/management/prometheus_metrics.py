from prometheus_client import Histogram

cms_doc_annotations = Histogram("cms_doc_annotations", "Number of annotations extracted from a document", ["handler"])
