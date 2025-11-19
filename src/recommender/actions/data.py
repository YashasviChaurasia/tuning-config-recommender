from recommender.utils.data_processing import load_training_data, escape_newlines_in_strings
from recommender.utils.data_config import (
    has_any_key_containing,
    determine_input_and_response_text,
    fetch_chat_template,
)
from .actions import Action, IR, PatchLevel, PatchType, Comment


class ApplyDataFormat(Action):
    def _is_data_in_required_format(self, sample: dict) -> bool:
        raise NotImplementedError

    def heuristic_skip(self, ir):
        path = ir.train_config.get("training_data_path")
        if not path:
            return True
        self.data = load_training_data(path)
        self.sample = self.data[0]
        return False


class ApplyQAFormat(ApplyDataFormat):

    COMMON_INPUT = ["instruction", "prompt", "question", "input", "query", "source"]
    COMMON_OUTPUT = ["output", "answer", "response", "label", "target", "completion"]

    def _is_data_in_required_format(self, sample: dict) -> bool:
        return (
            has_any_key_containing(sample, self.COMMON_INPUT)
            and has_any_key_containing(sample, self.COMMON_OUTPUT)
        )

    def apply(self, ir: IR) -> IR:
        self.skip = False
        if self.heuristic_skip(ir) or self.skip:
            return

        if not self._is_data_in_required_format(self.sample):
            self.skip = True
            return

        path = ir.train_config["training_data_path"]
        input_col, output_col = determine_input_and_response_text(path)
        template = f"\"### Input: {{{{ {input_col} }}}}\\n\\n### Response: {{{{ {output_col} }}}}\""

        ir.data_preprocessor = {
            "dataprocessor": {"type": "default", "streaming": False},
            "datasets": [
                {
                    "name": "dataset_from_inputs",
                    "data_paths": [path],
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
            ],
            "_dataset_text_field": "formatted_qa_data",
        }

        ir.comment = Comment("QA dataset formatting applied.")
        ir.type = PatchType.COMPATIBILITY
        ir.level = PatchLevel.MANDATORY
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir


class ApplyChatFormat(ApplyDataFormat):

    CHAT_KEYS = ["messages", "conversations", "dialogues", "chat", "turns"]

    def _is_data_in_required_format(self, sample: dict) -> bool:
        if not has_any_key_containing(sample, self.CHAT_KEYS):
            return False
        for k in self.CHAT_KEYS:
            if k in sample and isinstance(sample[k], list):
                return all(
                    isinstance(m, dict)
                    and "role" in m
                    and "content" in m
                    for m in sample[k]
                )
        return False

    def apply(self, ir: IR) -> IR:
        self.skip = False
        if self.heuristic_skip(ir) or self.skip:
            return

        if not self._is_data_in_required_format(self.sample):
            self.skip = True
            return

        path = ir.train_config["training_data_path"]

        chat_template, _ = fetch_chat_template(ir.train_config["model_name_or_path"])
        if chat_template:
            chat_template = escape_newlines_in_strings(chat_template)
            chat_template = f"{{% raw %}}\n{chat_template}\n{{% endraw %}}"

        conv_col = next(k for k in self.CHAT_KEYS if k in self.sample)

        ir.data_preprocessor = {
            "dataprocessor": {"type": "default", "streaming": False},
            "chat_template": chat_template,
            "datasets": [
                {
                    "name": "dataset_from_inputs",
                    "data_paths": [path],
                    "data_handlers": [
                        {
                            "name": "tokenize_and_apply_chat_template_with_masking",
                            "arguments": {
                                "remove_columns": "all",
                                "fn_kwargs": {
                                    "max_seq_length": ir.train_config.get("max_seq_length", 2048),
                                    "conversation_column_name": conv_col,
                                },
                            },
                        }
                    ],
                }
            ],
            "_conversation_column_name": conv_col,
        }

        ir.comment = Comment("Chat dataset formatting applied.")
        ir.type = PatchType.COMPATIBILITY
        ir.level = PatchLevel.MANDATORY
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir
