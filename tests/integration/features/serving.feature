Feature:
  CogStack ModelServe APIs

  Scenario Outline: Get general information about server healthiness, readiness and the running model
    Given CMS app is up and running
    When I send a GET request to <endpoint>
    Then the response should contain body <body> and status code <status_code>

    Examples:
      | endpoint  | body                        | status_code |
      | /healthz  | OK                          | 200         |
      | /readyz   | medcat_umls                 | 200         |
      | /info     | "model_type":"medcat_umls"  | 200         |

  Scenario: Extract entities from free texts
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data             | content_type |
      | /process        | Spinal stenosis  | text/plain   |
    Then the response should contain annotations

  Scenario: Extract entities from JSON Lines
    Given CMS app is up and running
    When I send a POST request with the following jsonlines content
      | endpoint        | data                                                                                            | content_type          |
      | /process_jsonl  | {"name": "doc1", "text": "Spinal stenosis"}\n{"name": "doc2", "text": "Spinal stenosis"}        | application/x-ndjson  |
    Then the response should contain json lines

  @skip
  Scenario: Extract entities from bulk texts
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data                                        | content_type          |
      | /process_bulk   | ["Spinal stenosis", "Spinal stenosis"]      | application/json      |
    Then the response should contain bulk annotations

  @skip
  Scenario: Extract entities from a file with bulk texts
    Given CMS app is up and running
    When I send a POST request with the following content where data as a file
      | endpoint             | data                                        | content_type          |
      | /process_bulk_file   | ["Spinal stenosis", "Spinal stenosis"]      | multipart/form-data   |
    Then the response should contain bulk annotations

  Scenario: Extract and redact entities from free texts
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data             | content_type |
      | /redact         | Spinal stenosis  | text/plain   |
    Then the response should contain text [spinal stenosis]

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

  Scenario: Warn when no entities are detected for redaction
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint                          | data            | content_type |
      | /redact?warn_on_no_redaction=true | abcdefgh  | text/plain   |
    Then the response should contain text warning: no entities were detected for redaction.

  Scenario: Extract and redact entities if not filtered out
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint                          | data             | content_type |
      | /redact?concepts_to_keep=C0037944 | Spinal stenosis  | text/plain   |
    Then the response should contain text spinal stenosis

  Scenario: Extract and redact entities with encryption
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint                | data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |  content_type       |
      | /redact_with_encryption | {"text": "Spinal stenosis", "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ITkTP8Tm/5FygcwY2EQ7LgVsuCF0OH7psUqvlXnOPNCfX86CobHBiSFjG9o5ZeajPtTXaf1thUodgpJZVZSqpVTXwGKo8r0COMO87IcwYigkZZgG/WmZgoZART+AA0+JvjFGxflJAxSv7puGlf82E+u5Wz2psLBSDO5qrnmaDZTvPh5eX84cocahVVI7X09/kI+sZiKauM69yoy1bdx16YIIeNm0M9qqS3tTrjouQiJfZ8jUKSZ44Na/81LMVw5O46+5GvwD+OsR43kQ0TexMwgtHxQQsiXLWHCDNy2ZzkzukDYRwA3V2lwVjtQN0WjxHg24BTBDBM+v7iQ7cbweQIDAQAB\n-----END PUBLIC KEY-----"} |  application/json   |
    Then the response should contain encrypted labels

  Scenario: Extract and preview entities
    Given CMS app is up and running
    When I send a POST request with the following content
      | endpoint        | data             | content_type |
      | /preview        | Spinal stenosis  | text/plain   |
    Then the response should contain a preview page

  Scenario: Preview trainer export
    Given CMS app is up and running
    When I send a POST request with the following trainer export
      | endpoint                                                | file_name           | content_type        |
      | /preview_trainer_export?project_id=14&document_id=3204  | trainer_export.json | multipart/form-data |
    Then the response should contain a preview page

  Scenario: Train supervised
    Given CMS app is up and running
    When I send a POST request with the following trainer export
      | endpoint                                                                                             | file_name           | content_type        |
      | /train_supervised?epochs=1&lr_override=0.01&test_size=0.2&early_stopping_patience=-1&log_frequency=1 | trainer_export.json | multipart/form-data |
    Then the response should contain the training ID
    When I send a GET request to /train_eval_info with that ID
    Then the response should contain the training information
    When I send a GET request to /train_eval_metrics with that ID
    Then the response should contain the supervised evaluation metrics

  Scenario: Train unsupervised
    Given CMS app is up and running
    When I send a POST request with the following training data
      | endpoint                                                    | file_name           | content_type        |
      | /train_unsupervised??epochs=1&test_size=0.2&log_frequency=1 | sample_texts.json   | multipart/form-data |
    Then the response should contain the training ID
    When I send a GET request to /train_eval_info with that ID
    Then the response should contain the training information
    When I send a GET request to /train_eval_metrics with that ID
    Then the response should contain the unsupervised evaluation metrics

  Scenario: Evaluate served model
    Given CMS app is up and running
    When I send a POST request with the following trainer export
      | endpoint  | file_name           | content_type        |
      | /evaluate | trainer_export.json | multipart/form-data |
    Then the response should contain the evaluation ID
    When I send a GET request to /train_eval_info with that ID
    Then the response should contain the evaluation information
