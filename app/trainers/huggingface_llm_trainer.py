import os
import logging
import math
import torch
import gc
import datasets
import re
import threading
import json
import inspect
import pandas as pd
from typing import final, Dict, TextIO, Optional, Any, List, Tuple, TYPE_CHECKING, Callable
from transformers import __version__ as transformers_version
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model # type: ignore
from app.management.model_manager import ModelManager
from app.management.tracker_client import TrackerClient
from app.utils import (
    reset_random_seed,
    non_default_device_is_available,
    create_model_data_package,
    get_model_data_package_extension,
    load_pydantic_object_from_dict,
    get_default_chat_template,
    get_default_system_prompt,
    get_model_data_package_base_name,
)
from app.trainers.base import SupervisedTrainer
from app.domain import ModelType, TrainerBackend, LlmRole, LlmTrainerType, LlmDatasetType, PromptMessage
from app.exception import (
    TrainingCancelledException,
    DatasetException,
    ConfigurationException,
    ExtraDependencyRequiredException,
)
if TYPE_CHECKING:
    from app.model_services.huggingface_llm_model import HuggingFaceLlmModel

logger = logging.getLogger("cms")


class _HuggingFaceLlmTrainerCommon(object):

    @staticmethod
    def deploy_model(
        model_service: "HuggingFaceLlmModel",
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        del model_service.model
        del model_service.tokenizer
        gc.collect()
        model_service.model = model
        model_service.tokenizer = tokenizer
        logger.info("Retrained model deployed")


@final
class HuggingFaceLlmSupervisedTrainer(SupervisedTrainer, _HuggingFaceLlmTrainerCommon):
    """
    A supervised trainer class for HuggingFace LLM models.

    Args:
        model_service (HuggingFaceLlmModel): An instance of the HuggingFace LLM model service.
    """

    MIN_EXAMPLE_COUNT_FOR_TRAINABLE_CONCEPT = 5
    MAX_CONCEPTS_TO_TRACK = 20
    PAD_LABEL_ID = -100
    DEFAULT_LABEL_ID = 0
    CONTINUING_TOKEN_LABEL_ID = 1

    def __init__(self, model_service: "HuggingFaceLlmModel") -> None:
        SupervisedTrainer.__init__(self, model_service._config, model_service.model_name)
        self._model_service = model_service
        self._model_name = model_service.model_name
        self._model_pack_path = model_service._model_pack_path
        self._retrained_models_dir = os.path.join(
            model_service._model_parent_dir,
            "retrained",
            self._model_name.replace(" ", "_"),
        )
        self._model_manager = ModelManager(type(model_service), model_service._config)
        self._max_length = model_service.model.config.max_position_embeddings
        os.makedirs(self._retrained_models_dir, exist_ok=True)

    def _load_dataset_from_config(self, data_file: TextIO, training_params: Dict) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """
        Loads training and validation datasets based on configuration in training_params.

        Args:
            data_file: The training data file
            training_params: Dictionary containing dataset configuration

        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        dataset_type = training_params.get("dataset_type", "json")

        # if dataset_type == "huggingface":
        #     return self._load_huggingface_dataset(training_params)
        if dataset_type == LlmDatasetType.JSON.value:
            return self._load_json_dataset(data_file, training_params)
        elif dataset_type == LlmDatasetType.CSV.value:
            return self._load_csv_dataset(data_file, training_params)
        else:
            raise DatasetException(f"Unsupported dataset type: {dataset_type}")

    @staticmethod
    def _set_dataset_format(train_dataset: datasets.Dataset, test_dataset: datasets.Dataset) -> None:
        """Sets the format of the datasets based on the dataset structure."""

        if "messages" in train_dataset.column_names:
            train_dataset.set_format(type=None, columns=["messages"])
            test_dataset.set_format(type=None, columns=["messages"])
        elif "question" in train_dataset.column_names and "answer" in train_dataset.column_names:
            train_dataset.set_format(type=None, columns=["question", "answer"])
            test_dataset.set_format(type=None, columns=["question", "answer"])
        elif "input" in train_dataset.column_names and "output" in train_dataset.column_names:
            train_dataset.set_format(type=None, columns=["input", "output"])
            test_dataset.set_format(type=None, columns=["input", "output"])
        elif "prompt" in train_dataset.column_names and "completion" in train_dataset.column_names:
            train_dataset.set_format(type=None, columns=["prompt", "completion"])
            test_dataset.set_format(type=None, columns=["prompt", "completion"])
        elif "problem" in train_dataset.column_names and "solution" in train_dataset.column_names:
            train_dataset.set_format(type=None, columns=["problem", "solution"])
            test_dataset.set_format(type=None, columns=["problem", "solution"])
        else:
            raise DatasetException("Unsupported dataset format")

    def _load_huggingface_dataset(self, training_params: Dict) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Loads dataset from HuggingFace Hub."""

        dataset_id = training_params.get("dataset_id", "AI-MO/NuminaMath-TIR")
        test_size = 0.2 if training_params.get("test_size") is None else training_params["test_size"]
        split_ratio = 1 - test_size
        train_percentage = int(split_ratio * 100)
        test_percentage = 100 - train_percentage
        train_split = training_params.get("train_split", f"train[:{train_percentage}%]")
        test_split = training_params.get("test_split", f"test[:{test_percentage}%]")

        logger.info(f"Loading HuggingFace dataset: {dataset_id}")
        train_dataset, test_dataset = datasets.load_dataset(dataset_id, split=[train_split, test_split])
        self._set_dataset_format(train_dataset, test_dataset)

        return train_dataset, test_dataset


    def _load_json_dataset(self, data_file: TextIO, training_params: Dict) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Loads dataset from JSON file."""

        data = json.load(data_file)
        test_size = 0.2 if training_params.get("test_size") is None else training_params["test_size"]
        split_ratio = 1 - test_size

        if isinstance(data, list):
            examples = data
            split_idx = int(len(examples) * split_ratio)
            train_examples = examples[:split_idx]
            test_examples = examples[split_idx:]
        elif isinstance(data, dict) and "train" in data and "test" in data:
            train_examples = data["train"]
            test_examples = data["test"]
        elif isinstance(data, dict) and "examples" in data:
            examples = data["examples"]
            split_idx = int(len(examples) * split_ratio)
            train_examples = examples[:split_idx]
            test_examples = examples[split_idx:]
        else:
            raise DatasetException("Unsupported JSON format")

        train_dataset = datasets.Dataset.from_list(train_examples)
        test_dataset = datasets.Dataset.from_list(test_examples)
        self._set_dataset_format(train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _load_csv_dataset(self, data_file: TextIO, training_params: Dict) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Loads dataset from CSV file."""

        df = pd.read_csv(data_file)
        test_size = 0.2 if training_params.get("test_size") is None else training_params["test_size"]
        split_ratio = 1 - test_size
        split_idx = int(len(df) * split_ratio)

        train_df = df[:split_idx]
        test_df = df[split_idx:]

        train_dataset = datasets.Dataset.from_pandas(train_df)
        test_dataset = datasets.Dataset.from_pandas(test_df)
        self._set_dataset_format(train_dataset, test_dataset)

        return train_dataset, test_dataset

    def _create_conversation_formatter(self, training_params: Dict) -> Callable:
        """
        Creates a conversation formatter based on training parameters.

        Args:
            training_params: Dictionary containing formatting configuration

        Returns:
            Function that formats examples into conversations
        """
        format_config = training_params.get("format_config", {})
        system_prompt = format_config.get("system_prompt", get_default_system_prompt())

        def make_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
            # Handle different input formats
            if "messages" in example:
                system_content = None
                question_content = None
                answer_content = None
                for message in example.get("messages", []):
                    msg = load_pydantic_object_from_dict(PromptMessage, message)
                    if msg.role == LlmRole.SYSTEM:
                        system_content = msg.content
                    elif msg.role == LlmRole.USER:
                        question_content = msg.content
                    elif msg.role == LlmRole.ASSISTANT:
                        answer_content = msg.content

                return {
                    "prompt": [
                        {"role": "system", "content": system_prompt if system_content is None else system_content},
                        {"role": "user", "content": question_content if question_content is not None else ""},
                    ],
                    "answer": answer_content if answer_content is not None else "",
                }
            elif "question" in example and "answer" in example:
                # Question/Answer format
                return {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": example.get("question")},
                    ],
                    "answer": example["answer"],
                }
            elif "input" in example and "output" in example:
                # Input/Output format
                return {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": example.get("input")},
                    ],
                    "answer": example["output"],
                }
            elif "prompt" in example and "completion" in example:
                # Prompt/Completion format
                return {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": example.get("prompt")},
                    ],
                    "answer": example["completion"],
                }
            elif "problem" in example and "solution" in example:
                # Problem/Solution format
                return {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": example.get("problem")},
                    ],
                    "answer": example["solution"],
                }
            else:
                raise DatasetException(f"Cannot determine the conversation format from example: {example}")

        return make_conversation

    def run(
        self,
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Runs the supervised training loop for HuggingFace LLM models.

        Args:
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (Union[TextIO, tempfile.TemporaryDirectory]): The file-like object or temporary directory containing the training data.
            log_frequency (int): The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id (str): The run ID of the training job.
            description (Optional[str]): The optional description of the training or change logs.
        """
        try:
            from trl import GRPOConfig, GRPOTrainer  # , PPOConfig, PPOTrainer
        except ImportError as e:
            logger.exception(e)
            logger.error("Cannot import the GRPO Trainer. Please install it with `pip install cms[llm]`.")
            raise ExtraDependencyRequiredException("Cannot import the GRPO Trainer. Please install it with `pip install cms[llm]`.")

        trained_model_pack_path = None
        redeploy = self._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = self._config.SKIP_SAVE_MODEL == "true"
        results_path = os.path.abspath(os.path.join(self._config.TRAINING_CACHE_DIR, "results"))
        logs_path = os.path.abspath(os.path.join(self._config.TRAINING_CACHE_DIR, "logs"))
        reset_random_seed()
        eval_mode = training_params["nepochs"] == 0
        self._tracker_client.log_trainer_mode(not eval_mode)
        trainer = None
        max_seq_length = 1024

        if not eval_mode:
            try:
                logger.info("Loading a PEFT model for training...")
                model_pack_file_ext = get_model_data_package_extension(self._model_pack_path)
                trained_model_pack_path = self._model_pack_path.replace(
                    model_pack_file_ext,
                    f"_trained_{run_id}{model_pack_file_ext}",
                )
                model, tokenizer = self._model_service.model, self._model_service.tokenizer
                trained_model_directory = os.path.join(
                    os.path.dirname(trained_model_pack_path),
                    get_model_data_package_base_name(trained_model_pack_path),
                )

                if non_default_device_is_available(self._config.DEVICE):
                    model.to(self._config.DEVICE)

                train_dataset, test_dataset = self._load_dataset_from_config(data_file, training_params)
                make_conversation = self._create_conversation_formatter(training_params)
                train_dataset = train_dataset.map(make_conversation)
                test_dataset = test_dataset.map(make_conversation)

                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                    logger.warning("The tokenizer does not have a chat template. Using the default one.")
                    tokenizer.chat_template = get_default_chat_template()
                else:
                    logger.debug(f"Found a chat template in the tokenizer:\n {tokenizer.chat_template}")

                lora_config = LoraConfig(
                    task_type="CAUSAL_LM",
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                )

                peft_model = get_peft_model(model, lora_config)

                mlflow_logging_callback = MLflowLoggingCallback(self._tracker_client)
                cancel_event_check_callback = CancelEventCheckCallback(self._cancel_event)
                trainer_callbacks = [mlflow_logging_callback, cancel_event_check_callback]

                trainer_type = training_params.get("trainer_type", LlmTrainerType.GRPO.value).lower()
                max_prompt_length = max(train_dataset.map(
                    lambda x: {
                        "tokens": tokenizer.apply_chat_template(
                            x["prompt"],
                            add_generation_prompt=True,
                            tokenize=True
                        )
                    },
                    batched=True,
                ).map(lambda x: {"length": len(x["tokens"])})["length"]) + 1
                if trainer_type == LlmTrainerType.PPO.value:
                    raise NotImplementedError("PPO training is not yet supported for HuggingFace LLM models")
                elif trainer_type == LlmTrainerType.GRPO.value:
                    training_args = GRPOConfig(
                        output_dir=results_path,
                        logging_dir=logs_path,
                        logging_steps=log_frequency,
                        learning_rate=5e-6,
                        adam_beta1=0.9,
                        adam_beta2=0.99,
                        weight_decay=0.1,
                        warmup_ratio=0.1,
                        lr_scheduler_type="cosine",
                        optim="paged_adamw_8bit",
                        per_device_train_batch_size=6,   # This global batch size must be divisible by the number of generations
                        gradient_accumulation_steps=1,
                        num_generations=6,
                        max_prompt_length=max_prompt_length,
                        max_completion_length=max_seq_length - max_prompt_length,
                        num_train_epochs = training_params["nepochs"],
                        save_steps=250,
                        max_grad_norm=0.1,
                        report_to="none",
                    )
                    trainer = GRPOTrainer(
                        model=peft_model,
                        processing_class=tokenizer,
                        reward_funcs=self._get_reward_functions(),
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=test_dataset,
                        callbacks=trainer_callbacks,
                    )
                else:
                    raise ConfigurationException(f"Unsupported trainer type: {trainer_type}")

                self._tracker_client.log_model_config({**model.config.to_dict(), **peft_model.peft_config})
                self._tracker_client.log_trainer_version(TrainerBackend.TRANSFORMERS, transformers_version)

                logger.info(f"Performing {trainer_type.upper()} training...")
                trainer.train()

                if cancel_event_check_callback.training_cancelled:
                    raise TrainingCancelledException("Training was cancelled by the user")

                if not skip_save_model:
                    model_pack_file_ext = get_model_data_package_extension(self._config.BASE_MODEL_FILE)
                    model_pack_file_name = f"{ModelType.HUGGINGFACE_LLM.value}_{run_id}{model_pack_file_ext}"
                    retrained_model_pack_path = os.path.join(self._retrained_models_dir, model_pack_file_name)
                    model = peft_model.merge_and_unload()
                    model.save_pretrained(
                        trained_model_directory,
                        safe_serialization=(self._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"),
                    )
                    tokenizer.save_pretrained(trained_model_directory)
                    create_model_data_package(trained_model_directory, retrained_model_pack_path)
                    model_uri = self._tracker_client.save_model(
                        retrained_model_pack_path,
                        self._model_name,
                        self._model_manager,
                        self._model_service.info().model_type.value,
                    )
                    logger.info(f"Retrained model saved: {model_uri}")
                else:
                    logger.info("Skipped saving on the retrained model")
                if redeploy:
                    self.deploy_model(self._model_service, model, tokenizer)
                else:
                    del model
                    del tokenizer
                    gc.collect()
                    logger.info("Skipped deployment on the retrained model")
                logger.info("Supervised training finished")
                self._tracker_client.end_with_success()
            except TrainingCancelledException as e:
                logger.exception(e)
                logger.info("Supervised training was cancelled")
                del model
                gc.collect()
                self._tracker_client.end_with_interruption()
            except torch.OutOfMemoryError as e:
                logger.exception("Supervised training failed on CUDA OOM")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        try:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.reset_accumulated_memory_stats()
                        except Exception:
                            pass
                        torch.cuda.synchronize()
                except Exception:
                    pass
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            except Exception as e:
                logger.exception("Supervised training failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with self._training_lock:
                    self._training_in_progress = False
                self._clean_up_training_cache()
                self._housekeep_file(trained_model_pack_path)
                if trainer is not None:
                    del trainer
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            try:
                logger.info("Evaluating the running model...")
                include_rewards_metrics = training_params.get("include_rewards_metrics", False)
                model, tokenizer = self._model_service.model, self._model_service.tokenizer
                if non_default_device_is_available(self._config.DEVICE):
                    model.to(self._config.DEVICE)

                eval_dataset, _ = self._load_dataset_from_config(data_file, training_params)
                make_conversation = self._create_conversation_formatter(training_params)
                eval_dataset = eval_dataset.map(make_conversation)
                max_prompt_length = max(eval_dataset.map(
                    lambda x: {
                        "tokens": tokenizer.apply_chat_template(
                            x["prompt"],
                            add_generation_prompt=True,
                            tokenize=True
                        )
                    },
                    batched=True,
                ).map(lambda x: {"length": len(x["tokens"])})["length"]) + 1

                training_args = GRPOConfig(
                    output_dir=results_path,
                    logging_dir=logs_path,
                    logging_steps=log_frequency,
                    per_device_eval_batch_size=6,
                    num_generations=2,
                    max_prompt_length=max_prompt_length,
                    max_completion_length=max_seq_length - max_prompt_length,
                    num_train_epochs=training_params["nepochs"],
                    report_to="none",
                    do_train=False,
                    do_eval=True,
                )

                mlflow_logging_callback = MLflowLoggingCallback(self._tracker_client)
                cancel_event_check_callback = CancelEventCheckCallback(self._cancel_event)
                trainer_callbacks = [mlflow_logging_callback, cancel_event_check_callback]

                trainer = GRPOTrainer(
                    model=model,
                    processing_class=tokenizer,
                    args=training_args,
                    reward_funcs=self._get_reward_functions(),
                    train_dataset=None,
                    eval_dataset=eval_dataset,
                    callbacks=trainer_callbacks,
                )

                eval_metrics = trainer.evaluate()
                if "perplexity" not in eval_metrics and "eval_loss" in eval_metrics:
                    eval_metrics.update({"perplexity": math.exp(eval_metrics["eval_loss"])})
                logger.info(f"Evaluation metrics: {eval_metrics}")
                self._tracker_client.send_hf_metrics_logs(eval_metrics, 0)
                if include_rewards_metrics:
                    try:
                        reward_metrics = self._evaluate_with_rewards(
                            model=model,
                            tokenizer=tokenizer,
                            eval_dataset=eval_dataset,
                            max_new_tokens=training_args.max_completion_length,
                        )
                        if reward_metrics:
                            logger.info(f"Reward metrics: {reward_metrics}")
                            self._tracker_client.send_hf_metrics_logs(reward_metrics, 0)
                    except Exception as e:
                        logger.warning(f"Failed to compute reward-based metrics: {e}")
                self._tracker_client.end_with_success()
                logger.info("Model evaluation finished")
            except torch.OutOfMemoryError as e:
                logger.exception("Evaluation failed on CUDA OOM")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        try:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.reset_accumulated_memory_stats()
                        except Exception:
                            pass
                        torch.cuda.synchronize()
                except Exception:
                    pass
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            except Exception as e:
                logger.exception("Evaluation failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with self._training_lock:
                    self._training_in_progress = False
                self._clean_up_training_cache()
                if trainer is not None:
                    del trainer
                    gc.collect()
                    torch.cuda.empty_cache()

    @staticmethod
    def _get_reward_functions() -> List:

        def extract_xml_answer(text: str) -> str:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()

        # Reward functions
        def correctness_reward_func(
                prompts: List,
                completions: List,
                answer: List,
                **kwargs: Dict[str, Any]
        ) -> List[float]:
            responses = [completion[0]["content"] for completion in completions]
            q = prompts[0][-1]["content"]
            extracted_responses = [extract_xml_answer(r) for r in responses]
            logger.debug(
                "%s\nQuestion:\n%s\nAnswer:\n%s\nResponse:\n%s\nExtracted:\n%s",
                "-" * 20,
                q,
                answer[0],
                responses[0],
                extracted_responses[0]
            )
            return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

        def int_reward_func(completions: Tuple[Any], **kwargs: Dict[str, Any]) -> List[float]:
            responses = [completion[0]["content"] for completion in completions]
            extracted_responses = [extract_xml_answer(r) for r in responses]
            return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

        def strict_format_reward_func(completions: Tuple[Any], **kwargs: Dict[str, Any]) -> List[float]:
            """Reward function that checks if the completion has a specific format."""
            pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
            responses = [completion[0]["content"] for completion in completions]
            matches = [re.match(pattern, r) for r in responses]
            return [0.5 if match else 0.0 for match in matches]

        def soft_format_reward_func(completions: Tuple[Any], **kwargs: Dict[str, Any]) -> List[float]:
            """Reward function that checks if the completion has a specific format."""
            pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
            responses = [completion[0]["content"] for completion in completions]
            matches = [re.match(pattern, r) for r in responses]
            return [0.5 if match else 0.0 for match in matches]

        def count_xml(text: str) -> float:
            count = 0.0
            if text.count("<reasoning>\n") == 1:
                count += 0.125
            if text.count("\n</reasoning>\n") == 1:
                count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1]) * 0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
            return count

        def xmlcount_reward_func(completions: Tuple[Any], **kwargs: Dict[str, Any]) -> List[float]:
            contents = [completion[0]["content"] for completion in completions]
            return [count_xml(c) for c in contents]

        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]

    def _evaluate_with_rewards(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: datasets.Dataset,
        max_new_tokens: int,
    ) -> Dict[str, float]:
        model.eval()
        if non_default_device_is_available(self._config.DEVICE):
            model.to(self._config.DEVICE)

        reward_funcs = self._get_reward_functions()
        reward_sums: Dict[str, float] = {fn.__name__: 0.0 for fn in reward_funcs}
        count = 0

        for example in eval_dataset:
            if "prompt" not in example:
                continue
            messages = example["prompt"]
            answer = example.get("answer", "")

            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            if non_default_device_is_available(self._config.DEVICE):
                input_ids = input_ids.to(self._config.DEVICE)
                attention_mask = attention_mask.to(self._config.DEVICE)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    pad_token_id=getattr(tokenizer, "pad_token_id", 0),
                )

            completion_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
            for fn in reward_funcs:
                sig = inspect.signature(fn)
                kwargs: Dict[str, Any] = {}
                if "prompts" in sig.parameters:
                    kwargs["prompts"] = [messages]
                if "completions" in sig.parameters:
                    kwargs["completions"] = [({"content": completion_text},)]
                if "answer" in sig.parameters:
                    kwargs["answer"] = [answer]

                try:
                    rewards = fn(**kwargs)  # type: ignore
                    value = float(rewards[0]) if isinstance(rewards, (list, tuple)) and rewards else float(rewards)
                except Exception:
                    value = 0.0

                reward_sums[fn.__name__] += value
            count += 1
        if count == 0:
            return {}

        reward_avgs = {f"reward_{name}": total / count for name, total in reward_sums.items()}
        reward_overall_mean = sum(reward_avgs.values()) / len(reward_avgs) if reward_avgs else 0.0
        reward_avgs["reward_overall_mean"] = reward_overall_mean
        reward_avgs["reward_samples"] = float(count)
        return reward_avgs


@final
class MLflowLoggingCallback(TrainerCallback):
    """
    A callback class for logging training metrics to MLflow.

    Args:
        tracker_client (TrackerClient): An instance of TrackerClient used for logging.
    """

    def __init__(self, tracker_client: TrackerClient) -> None:
        self.tracker_client = tracker_client
        self.epoch = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Logs metrics at the end of each epoch.

        Args:
            args (TrainingArguments): The arguments used for training.
            state (TrainerState): The current state of the Trainer.
            control (TrainerControl): The current control of the Trainer.
            logs (Dict[str, float]): A dictionary containing the metrics to log.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        """

        if logs is not None:
            if logs.get("eval_loss", None) is not None:
                logs["perplexity"] = math.exp(logs["eval_loss"])
            self.tracker_client.send_hf_metrics_logs(logs, self.epoch)
        self.epoch += 1


@final
class CancelEventCheckCallback(TrainerCallback):
    """
    A callback class for checking a cancellation event during training.

    Args:
        cancel_event (threading.Event): A threading event that signals whether training should be cancelled.
    """

    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event
        self.training_cancelled = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Checks if the training should be cancelled at the end of each training step.

        Args:
            args (TrainingArguments): The arguments used for training.
            state (TrainerState): The current state of the Trainer.
            control (TrainerControl): The current control of the Trainer.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        """

        if self.cancel_event.is_set():
            control.should_training_stop = True
            self.cancel_event.clear()
            self.training_cancelled = True
