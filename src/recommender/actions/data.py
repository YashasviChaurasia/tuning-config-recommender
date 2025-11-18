from typing import Dict
from recommender.utils.data_processing import (
    load_training_data,
    escape_newlines_in_strings,
)
from recommender.utils.data_config import (
    has_any_key_containing,
    determine_input_and_response_text,
    fetch_chat_template,
)
from .actions import Action, IR, PatchLevel, PatchType, Comment


class ApplyDataFormat(Action):
    def heuristic_skip(self, ir):
        path = ir.train_config.get("training_data_path")
        return not path

    def _is_data_tokenized(self, path):
        data = load_training_data(path)
        if not data or not isinstance(data, list):
            return False
        if not isinstance(data[0], dict):
            return False
        tokenized_fields = {"input_ids", "labels", "attention_mask"}
        return any(f in data[0] for f in tokenized_fields)


class ApplyQAFormat(ApplyDataFormat):

    COMMON_INPUT_KEYS = [
        "input", "instruction", "prompt", "question",
        "tweet_text", "query", "source", "tweet text",
    ]
    COMMON_RESPONSE_KEYS = [
        "output", "response", "answer", "label",
        "text_label", "target", "completion",
    ]

    def _is_data_in_required_format(self, path: str) -> bool:
        try:
            data = load_training_data(path)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
        except:
            return False
        return (
            has_any_key_containing(data, self.COMMON_INPUT_KEYS)
            and has_any_key_containing(data, self.COMMON_RESPONSE_KEYS)
        )

    def _get_values_for_path(self, path: str):
        input_txt, response_txt = determine_input_and_response_text(path)

        template = (
            "### Input: {{%s}} \n\n### Response: {{%s}}"
            % (input_txt, response_txt)
        )

        return {
            "dataset_text_field": "formatted_qa_data",
            "response_template": "### Response:",
            "template": template,
            "data_handlers": [
                {
                    "name": "apply_custom_jinja_template",
                    "arguments": {
                        "remove_columns": "all",
                        "batched": False,
                        "fn_kwargs": {
                            "formatted_text_column_name": "formatted_qa_data",
                            "template": template,
                        },
                    },
                }
            ],
        }

    def apply(self, ir: IR) -> IR:
        if self.skip or self.heuristic_skip(ir):
            self.skip = True
            return

        path = ir.train_config["training_data_path"]

        ir.data_preprocessor["dataprocessor"] = {
            "type": "default",
            "streaming": False,
        }
        ir.data_preprocessor["datasets"] = [
            {
                "name": "dataset_from_inputs",
                "data_paths": [path],
                "data_handlers": {},
            }
        ]

        dataset = ir.data_preprocessor["datasets"][0]
        vals = self._get_values_for_path(path)

        dataset["data_handlers"] = vals["data_handlers"]
        dataset["dataset_text_field"] = vals["dataset_text_field"]
        dataset["response_template"] = vals["response_template"]
        dataset["template"] = vals["template"]

        try:
            sample = load_training_data(path)[0]
            keys = {k.lower() for k in sample}
            if "text" in keys and "label" in keys:
                dataset["column_mapping"] = {"input": "text", "target": "label"}
        except:
            pass

        ir.comment = Comment("This data config is used for formatting QA datasets.")
        ir.type = PatchType.COMPATIBILITY
        ir.level = PatchLevel.MANDATORY
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir


class ApplyChatFormat(ApplyDataFormat):
    CHAT_KEYS = ["messages", "conversations", "dialogues", "chat", "turns"]

    def _is_data_in_required_format(self, path: str) -> bool:
        try:
            data = load_training_data(path)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
        except:
            return False

        for k in self.CHAT_KEYS:
            if k in data and isinstance(data[k], list):
                if all(isinstance(m, dict) and "role" in m and "content" in m for m in data[k]):
                    return True

        return False

    def _get_values_for_path(self, path: str, model: str, max_len: int):
        chat_template, _ = fetch_chat_template(model)
        if chat_template:
            chat_template = escape_newlines_in_strings(chat_template)
            chat_template = "{% raw %}\n" + chat_template + "\n{% endraw %}"

        data = load_training_data(path)
        conv_col = None
        for k in self.CHAT_KEYS:
            if k in data:
                conv_col = k
                break

        if conv_col is None:
            return None

        return {
            "chat_template": chat_template,
            "conversation_column_name": conv_col,
            "data_handlers": [
                {
                    "name": "tokenize_and_apply_chat_template_with_masking",
                    "arguments": {
                        "remove_columns": "all",
                        "fn_kwargs": {
                            "max_seq_length": max_len,
                            "conversation_column_name": conv_col,
                        },
                    },
                }
            ],
        }

    def apply(self, ir: IR) -> IR:
        if self.skip or self.heuristic_skip(ir):
            self.skip = True
            return

        path = ir.train_config["training_data_path"]

        if not self._is_data_in_required_format(path):
            self.skip = True
            return

        ir.data_preprocessor["dataprocessor"] = {
            "type": "default",
            "streaming": False,
        }
        ir.data_preprocessor["datasets"] = [
            {
                "name": "dataset_from_inputs",
                "data_paths": [path],
                "data_handlers": {},
            }
        ]

        vals = self._get_values_for_path(
            path,
            ir.train_config["model_name_or_path"],
            ir.train_config.get("max_seq_length", 2048),
        )

        if vals is None:
            self.skip = True
            return

        dataset = ir.data_preprocessor["datasets"][0]
        dataset["data_handlers"] = vals["data_handlers"]
        ir.data_preprocessor["chat_template"] = vals["chat_template"]

        ir.comment = Comment("This data config is used for formatting chat datasets.")
        ir.type = PatchType.COMPATIBILITY
        ir.level = PatchLevel.MANDATORY
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir
