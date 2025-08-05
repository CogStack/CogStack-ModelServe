class PromptFactory:

    _ALPACA = (
        "{% if messages[0]['role'] == 'system' %}"
          "{% set loop_messages = messages[1:] %}"
          "{% set system_message = messages[0]['content'].strip() + '\n' %}"
        "{% else %}"
          "{% set loop_messages = messages %}"
          "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
          "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
          "{% endif %}"
          "{% if loop.index0 == 0 %}"
            "{% set content = system_message + message['content'] %}"
          "{% else %}"
            "{% set content = message['content'] %}"
          "{% endif %}"

          "{% if message['role'] == 'user' %}"
            "{{ '### Instruction:\n' + content.strip() + '\n\n'}}"
          "{% elif message['role'] == 'assistant' %}"
            "{{ '### Response:\n'  + content.strip() + '\n\n' }}"
          "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
          "{{ '### Response:\n' }}"
        "{% endif %}"
    )

    _CHAT_ML = (
        "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'].strip() + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
        "{% endif %}"
    )

    _DEFAULT = (
        "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
                "{{'<|user|>\n' + message['content'] + eos_token}}"
            "{% elif message['role'] == 'system' %}"
                "{{'<|system|>\n' + message['content'] + eos_token}}"
            "{% elif message['role'] == 'assistant' %}"
                "{{'<|assistant|>\n' + message['content'] + eos_token}}"
            "{% endif %}"
            "{% if loop.last and add_generation_prompt %}"
                "{{'<|assistant|>'}}"
            "{% endif %}"
        "{% endfor %}"
    )

    _FALCON = (
        "{% if messages[0]['role'] == 'system' %}"
          "{% set loop_messages = messages[1:] %}"
          "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
          "{% set loop_messages = messages %}"
          "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
          "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
          "{% endif %}"
          "{% if loop.index0 == 0 %}"
            "{{ system_message.strip() }}"
          "{% endif %}"
          "{{ '\n\n' + message['role'].title() + ': ' + message['content'].strip().replace('\r\n', '\n').replace('\n\n', '\n') }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
          "{ '\n\nAssistant:' }}"
        "{% endif %}"
    )

    _GEMMA = (
        "{% if messages[0]['role'] == 'system' %}"
          "{% set loop_messages = messages[1:] %}"
          "{% set system_message = messages[0]['content'].strip() + '\n\n' %}"
        "{% else %}"
          "{% set loop_messages = messages %}"
          "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
          "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
          "{% endif %}"
          "{% if loop.index0 == 0 %}"
            "{% set content = system_message + message['content'] %}"
          "{% else %}"
            "{% set content = message['content'] %}"
          "{% endif %}"
          "{% if (message['role'] == 'assistant') %}"
            "{% set role = 'model' %}"
          "{% else %}"
            "{% set role = message['role'] %}"
          "{% endif %}"
          "{{ '<start_of_turn>' + role + '\n' + content.strip() + '<end_of_turn>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
          "{{'<start_of_turn>model\n'}}"
        "{% endif %}"
    )

    _LLAMA_2 = (
        "{% if messages[0]['role'] == 'system' %}"
          "{% set loop_messages = messages[1:] %}"
          "{% set system_message = '<<SYS>>\n' + messages[0]['content'].strip() + '\n<</SYS>>\n\n' %}"
        "{% else %}"
          "{% set loop_messages = messages %}"
          "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
          "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
          "{% endif %}"
          "{% if loop.index0 == 0 %}"
            "{% set content = system_message + message['content'] %}"
          "{% else %}"
            "{% set content = message['content'] %}"
          "{% endif %}"
          "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
          "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
          "{% endif %}"
        "{% endfor %}"
    )

    _LLAMA_3 = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
          "{% set loop_messages = messages[1:] %}"
          "{% set system_message = '<|start_header_id|>' + 'system' + '<|end_header_id|>\n\n' + messages[0]['content'].strip() + '<|eot_id|>' %}"
        "{% else %}"
          "{% set loop_messages = messages %}"
          "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
          "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
          "{% endif %}"
          "{% if loop.index0 == 0 %}"
            "{{ system_message }}"
          "{% endif %}"
          "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'].strip() + '<|eot_id|>' }}"
          "{% if loop.last and message['role'] == 'user' and add_generation_prompt %}"
            "{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}"
          "{% endif %}"
        "{% endfor %}"
    )

    _MISTRAL = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
          "{% set loop_messages = messages[1:] %}"
          "{% set system_message = messages[0]['content'].strip() + '\n\n' %}"
        "{% else %}"
          "{% set loop_messages = messages %}"
          "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
          "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
          "{% endif %}"
          "{% if loop.index0 == 0 %}"
            "{% set content = system_message + message['content'] %}"
          "{% else %}"
            "{% set content = message['content'] %}"
          "{% endif %}"
          "{% if message['role'] == 'user' %}"
            "{{ '[INST] ' + content.strip() + ' [/INST]' }}"
          "{% elif message['role'] == 'assistant' %}"
            "{{ content.strip() + eos_token}}"
          "{% endif %}"
        "{% endfor %}"
    )

    _PHI_2 = (
        "{% if messages[0]['role'] == 'system' %}"
          "{% set loop_messages = messages[1:] %}"
          "{% set system_message = messages[0]['content'].strip() + '\n\n' %}"
        "{% else %}"
          "{% set loop_messages = messages %}"
          "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
          "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
          "{% endif %}"
          "{% if loop.index0 == 0 %}"
            "{% set content = system_message + message['content'] %}"
          "{% else %}"
            "{% set content = message['content'] %}"
          "{% endif %}"
          "{% if message['role'] == 'user' %}"
            "{{ 'Instruct: ' + content.strip() + '\n' }}"
          "{% elif message['role'] == 'assistant' %}"
            "{{ 'Output: '  + content.strip() + '\n' }}"
          "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
          "{{ 'Output:' }}"
        "{% endif %}"
    )

    _PHI_3 = (
        "{{ bos_token }}"
        "{% for message in messages %}"
            "{% if (message['role'] == 'system') %}"
                "{{'<|system|>' + '\n' + message['content'].strip() + '<|end|>' + '\n'}}"
            "{% elif (message['role'] == 'user') %}"
                "{{'<|user|>' + '\n' + message['content'].strip() + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}"
            "{% elif message['role'] == 'assistant' %}"
                "{{message['content'].strip() + '<|end|>' + '\n'}}"
            "{% endif %}"
        "{% endfor %}"
    )

    _QWEN = (
        "{% for message in messages %}"
          "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}"
          "{% endif %}"
          "{{'<|im_start|>' + message['role'] + '\n' + message['content'].strip() }}"
          "{% if (loop.last and add_generation_prompt) or not loop.last %}"
            "{{ '<|im_end|>' + '\n'}}"
          "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}"
          "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    @classmethod
    def create_chat_template(cls, name: str = "default") -> str:
        if name.lower() == "default":
            return cls._DEFAULT
        elif name.lower() == "alpaca":
            return cls._ALPACA
        elif name.lower() == "chat_ml":
            return cls._CHAT_ML
        elif name.lower() == "falcon":
            return cls._FALCON
        elif name.lower() == "gemma":
            return cls._GEMMA
        elif name.lower() == "llama_2":
            return cls._LLAMA_2
        elif name.lower() == "llama_3":
            return cls._LLAMA_3
        elif name.lower() == "mistral":
            return cls._MISTRAL
        elif name.lower() == "phi_2":
            return cls._PHI_2
        elif name.lower() == "phi_3":
            return cls._PHI_3
        elif name.lower() == "qwen":
            return cls._QWEN
        else:
            raise ValueError("Invalid template name")
