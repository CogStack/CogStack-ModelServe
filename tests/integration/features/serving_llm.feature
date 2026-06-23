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

  @openai-models
  Scenario: List OpenAI-compatible models
    Given CMS LLM app is up and running
    When I send a GET request to endpoint
      | endpoint          |
      | /openai/v1/models |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key    |
      | object |
      | data   |

  @openai-chat
  Scenario: Create OpenAI-compatible chat completion
    Given CMS LLM app is up and running
    When I send a POST request with JSON body
      | endpoint                    | body                                                                                                                                                                             |
      | /openai/v1/chat/completions | {"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is spinal stenosis?"}],"model":"test_model","stream":false} |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key     |
      | object  |
      | choices |

  @openai-completions
  Scenario: Create OpenAI-compatible completion
    Given CMS LLM app is up and running
    When I send a POST request with JSON body
      | endpoint               | body                                                                                 |
      | /openai/v1/completions | {"model":"test_model","prompt":"What is spinal stenosis?","stream":false} |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key     |
      | object  |
      | choices |

  @openai-embeddings
  Scenario: Create OpenAI-compatible embeddings
    Given CMS LLM app is up and running
    When I send a POST request with JSON body
      | endpoint              | body                                                          |
      | /openai/v1/embeddings | {"model":"test_model","input":["spinal stenosis"]} |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key    |
      | object |
      | data   |

  @ollama-tags
  Scenario: List Ollama-compatible tags
    Given CMS LLM app is up and running
    When I send a GET request to endpoint
      | endpoint         |
      | /ollama/api/tags |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key    |
      | models |

  @ollama-health-check @get
  Scenario: Ollama-compatible health check with GET
    Given CMS LLM app is up and running
    When I send a GET request to endpoint
      | endpoint |
      | /ollama/ |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key    |
      | status |

  @ollama-health-check @head
  Scenario: Ollama-compatible health check with HEAD
    Given CMS LLM app is up and running
    When I send a HEAD request to endpoint
      | endpoint |
      | /ollama/ |
    Then the response status code should be 200

  @ollama-version
  Scenario: Get Ollama-compatible API version
    Given CMS LLM app is up and running
    When I send a GET request to endpoint
      | endpoint            |
      | /ollama/api/version |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key     |
      | version |

  @ollama-show
  Scenario: Show Ollama-compatible model information
    Given CMS LLM app is up and running
    When I send a POST request with JSON body
      | endpoint         | body                                   |
      | /ollama/api/show | {"model":"test_model"}      |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key        |
      | modelfile  |
      | model_info |

  @ollama-generate
  Scenario: Create Ollama-compatible generation
    Given CMS LLM app is up and running
    When I send a POST request with JSON body
      | endpoint             | body                                                                                 |
      | /ollama/api/generate | {"model":"test_model","prompt":"What is spinal stenosis?","stream":false} |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key      |
      | response |
      | done     |

  @ollama-chat
  Scenario: Create Ollama-compatible chat completion
    Given CMS LLM app is up and running
    When I send a POST request with JSON body
      | endpoint         | body                                                                                                               |
      | /ollama/api/chat | {"model":"test_model","messages":[{"role":"user","content":"What is spinal stenosis?"}],"stream":false} |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key     |
      | message |
      | done    |

  @ollama-embed
  Scenario: Create Ollama-compatible embeddings
    Given CMS LLM app is up and running
    When I send a POST request with JSON body
      | endpoint          | body                                                          |
      | /ollama/api/embed | {"model":"test_model","input":["spinal stenosis"]} |
    Then the response status code should be 200
    And the response content type should contain application/json
    And the JSON response should include keys
      | key        |
      | embeddings |
