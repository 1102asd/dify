from typing import cast

from core.workflow.entities.base_node_data_entities import BaseNodeData
from core.workflow.entities.node_entities import NodeRunResult, NodeType
from core.workflow.entities.variable_pool import VariablePool
from core.workflow.nodes.base_node import BaseNode
from core.workflow.nodes.variable_assigner.entities import VariableAssignerNodeData
from models.workflow import WorkflowNodeExecutionStatus


class VariableAssignerNode(BaseNode):
    _node_data_cls = VariableAssignerNodeData
    _node_type = NodeType.VARIABLE_ASSIGNER

    def _run(self, variable_pool: VariablePool) -> NodeRunResult:
        node_data: VariableAssignerNodeData = cast(self._node_data_cls, self.node_data)
        outputs = {}
        for variable in node_data.variables:
            value = variable_pool.get_variable_value(variable)
            if value:
                variable_pool.append_variable(
                    node_id=self.node_id,
                    variable_key_list=variable,
                    value=value
                )
                outputs = {
                    "output": value
                }
                break

        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            outputs=outputs,
        )

    @classmethod
    def _extract_variable_selector_to_variable_mapping(cls, node_data: BaseNodeData) -> dict[str, list[str]]:
        return {}