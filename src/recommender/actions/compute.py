from .actions import IR, Action, Comment, PatchLevel, PatchType


class ApplyComputeConfig(Action):
    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        # TODO: fast kernels are not supported for some optimizer classes
        # we should either edit this or skip this optimization
        num_nodes = ir.compute_config.get("num_nodes", 1)
        num_gpus_per_node = ir.compute_config.get("num_gpus_per_node", 8)

        return_ir = IR(
            compute_config={
                "num_nodes": num_nodes,
                "num_gpus_per_node": num_gpus_per_node,
            },
            type=PatchType.COMPATIBILITY,
            level=PatchLevel.SUGGESTION,
            comment=Comment(
                "compute config for single node configuration"
                if (num_nodes == 1 and num_gpus_per_node == 8)
                else ""
            ),
        )
        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir
