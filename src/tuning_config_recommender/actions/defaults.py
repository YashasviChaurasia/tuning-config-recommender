from .actions import IR, Action, Comment, PatchLevel, PatchType


class ApplyDefaults(Action):
    def apply(self, ir: IR, actions_meta: list[str]) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        return_ir = IR(
            tuning_config={
                "logging_steps": 1,
                "logging_strategy": "steps",
                "dataloader_drop_last": True,
                "bf16": "True",
                "ddp_timeout": "7200",
                "warmup_ratio": 0.03,
                "lr_scheduler_type": "linear",
                "learning_rate": "1e-06",
                "warmup_steps": 200,
                "adam_beta1": 0.9,
                "adam_beta2": 0.98,
                "weight_decay": 0.1,
                "adam_epsilon": 1e-10,
            },
            type=PatchType.MODEL_QUALITY,
            level=PatchLevel.SUGGESTION,
            comment=Comment(
                "These are bunch of defaults which impact model quality and performance."
            ),
        )
        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir
