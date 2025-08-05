from jinja2.sandbox import ImmutableSandboxedEnvironment
from app.processors.prompt_factory import PromptFactory


def test_create_default_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("default"))
    prompt = template.render(
        messages=messages,
        bos_token="<|system|>",
        eos_token="<|end|>",
        add_generation_prompt=True,
    )
    assert prompt == "<|system|>\nAlright?<|end|><|user|>\nYeah.<|end|><|assistant|>"


def test_create_alpaca_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("alpaca"))
    prompt = template.render(
        messages=messages,
        add_generation_prompt=True,
    )
    assert prompt == "### Instruction:\nAlright?\nYeah.\n\n### Response:\n"


def test_create_chat_ml_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("chat_ml"))
    prompt = template.render(
        messages=messages,
        add_generation_prompt=True,
    )
    assert prompt == "<|im_start|>system\nAlright?<|im_end|>\n<|im_start|>user\nYeah.<|im_end|>\n<|im_start|>assistant\n"

def test_create_falcon_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("falcon"))
    prompt = template.render(
        messages=messages,
        add_generation_prompt=True,
    )
    assert prompt == "Alright?\n\nUser: Yeah.{ '\n\nAssistant:' }}"

def test_create_gemma_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("gemma"))
    prompt = template.render(
        messages=messages,
        add_generation_prompt=True,
    )
    assert prompt == "<start_of_turn>user\nAlright?\n\nYeah.<end_of_turn>\n<start_of_turn>model\n"

def test_create_llama_2_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("LLAMA_2"))
    prompt = template.render(
        messages=messages,
        bos_token="<s>",
        eos_token="</s>",
        add_generation_prompt=True,
    )
    assert prompt == "<s>[INST] <<SYS>>\nAlright?\n<</SYS>>\n\nYeah. [/INST]"

def test_create_llama_3_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("LLAMA_2"))
    prompt = template.render(
        messages=messages,
        bos_token="<s>",
        eos_token="</s>",
        add_generation_prompt=True,
    )
    assert prompt == "<s>[INST] <<SYS>>\nAlright?\n<</SYS>>\n\nYeah. [/INST]"

def test_create_mistral_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("mistral"))
    prompt = template.render(
        messages=messages,
        bos_token="<s>",
        eos_token="</s>",
        add_generation_prompt=True,
    )
    assert prompt == "<s>[INST] Alright?\n\nYeah. [/INST]"

def test_create_phi_2_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("phi_2"))
    prompt = template.render(
        messages=messages,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_generation_prompt=True,
    )
    assert prompt == "Instruct: Alright?\n\nYeah.\nOutput:"

def test_create_phi_3_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("phi_3"))
    prompt = template.render(
        messages=messages,
        bos_token="<s>",
        eos_token="<|end|>",
        add_generation_prompt=True,
    )
    assert prompt == "<s><|system|>\nAlright?<|end|>\n<|user|>\nYeah.<|end|>\n<|assistant|>\n"

def test_create_qwen_chat_template():
    env = ImmutableSandboxedEnvironment()
    messages = [
        {
            "role": "system",
            "content": "Alright?"
        },
        {
            "role": "user",
            "content": "Yeah."
        },
    ]
    template = env.from_string(PromptFactory.create_chat_template("qwen"))
    prompt = template.render(
        messages=messages,
        bos_token="<s>",
        eos_token="<|end|>",
        add_generation_prompt=True,
    )
    assert prompt == "<|im_start|>system\nAlright?<|im_end|>\n<|im_start|>user\nYeah.<|im_end|>\n<|im_start|>assistant\n"
