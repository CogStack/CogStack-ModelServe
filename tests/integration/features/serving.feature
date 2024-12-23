Feature:
  CogStack ModelServe APIs

  Scenario Outline: Get general information about server healthiness, readiness and the running model
    Given CMS app is up and running
    When I send a GET request to <endpoint>
    Then the response should contain body <body> and status code <status_code>

    Examples:
      | endpoint  | body                                                                                                                  | status_code |
      | /healthz  | OK                                                                                                                    | 200         |
      | /readyz   | medcat_snomed                                                                                                         | 200         |
      | /info     | {"api_version":"0.0.1","model_type":"medcat_snomed","model_description":"medcat_model_description","model_card":null} | 200         |

  Scenario: Extract entities from free texts
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data             | content_type |
      | /process        | Spinal stenosis  | text/plain   |
    Then the response should contain annotations

  Scenario: Extract entities from JSON Lines
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data                                                                                            | content_type          |
      | /process_jsonl  | {"name": "doc1", "text": "Spinal stenosis"}\n{"name": "doc2", "text": "Spinal stenosis"}        | application/x-ndjson  |
    Then the response should contain json lines

  Scenario: Extract entities from bulk texts
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data                                        | content_type          |
      | /process_bulk   | ["Spinal stenosis", "Spinal stenosis"]      | application/json      |
    Then the response should contain bulk annotations

  Scenario: Extract and redact entities from free texts
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data             | content_type |
      | /redact         | Spinal stenosis  | text/plain   |
    Then the response should contain text [Spinal stenosis]

  Scenario: Extract and redact entities from free texts with a mask
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint          | data             | content_type |
      | /redact?mask=***  | Spinal stenosis  | text/plain   |
    Then the response should contain text ***

  Scenario: Extract and redact entities from free texts with a hash
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint                    | data             | content_type |
      | /redact?mask=any&hash=true  | Spinal stenosis  | text/plain   |
    Then the response should contain text 4c86af83314100034ad83fae3227e595fc54cb864c69ea912cd5290b8d0f41a4

  Scenario: Extract and preview entities
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data             | content_type |
      | /preview        | Spinal stenosis  | text/plain   |
    Then the response should contain a preview page
