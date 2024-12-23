Feature:
  CogStack ModelServe Stream APIs

Scenario: Stream entities extracted from free texts
    Given CMS stream app is up and running
    When I send an async POST request with the following content
      | endpoint        | data                                                                                      | content_type          |
      | /stream/process | {"name": "doc1", "text": "Spinal stenosis"}\n{"name": "doc2", "text": "Spinal stenosis"}  | application/x-ndjson  |
    Then the response should contain annotation stream

Scenario: Interactively extract entities from free texts
    Given CMS stream app is up and running
    When I send a piece of text to the WS endpoint
    Then the response should contain annotated spans
