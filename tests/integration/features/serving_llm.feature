Feature:
  CogStack ModelServe APIs (LLM)

  @generate
  Scenario: Generate text from a prompt
    Given CMS LLM app is up and running
    When I send a POST request with the following prompt
      | endpoint  | prompt                   | content_type |
      | /generate | What is spinal stenosis? | text/plain   |
    Then the response should contain generated text

  @generate-stream
  Scenario: Generate text stream from a prompt
    Given CMS LLM app is up and running
    When I send a POST request with the following prompt
      | endpoint         | prompt                   | content_type |
      | /stream/generate | What is spinal stenosis? | text/plain   |
    Then the response should contain generated text stream