"""Data Permission（架构文档 §40，任务 #53）。

RAG 检索的 metadata filter：

    tenant_id = <租户>
    AND array_contains(role_scope, <角色>)
    AND (orchard_scope 含 "ALL" 或含 <用户园区>)

- VectorIndex 的 schema 自 #25 起已带 role_scope/orchard_scope 数组字段，
  本模块补齐 tenant 维 + 统一的表达式构造，并接入检索链路
  （图 retrieve 节点与 AgricultureRagTool）；
- orchard_scope 为空 = 全园区可见：**索引侧写入哨兵 "ALL"**
  （Milvus 表达式判空数组不可靠，哨兵化后统一 array_contains）；
- admin 不做 role 过滤（管理面全量）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

#: 全园区哨兵（索引侧：空 orchard_scope 写为 ["ALL"]）
ALL_ORCHARDS = "ALL"


@dataclass
class DataScope:
    tenant_id: str = "default"
    role: str = "worker"
    orchard_scope: List[str] = field(default_factory=list)   # 空=全部园区


def normalize_orchard_scope(scope: Optional[List[str]]) -> List[str]:
    """索引/查询两侧统一：空 → ["ALL"]。"""
    cleaned = [s for s in (scope or []) if s]
    return cleaned or [ALL_ORCHARDS]


def build_scope_expr(scope: DataScope) -> str:
    """scope → Milvus filter 表达式（§40 三条件联合）。

    用户 orchard_scope 为空 = 全部园区权限（不加园区过滤，能看受限文档）；
    非空 = 限指定园区（只能看全园区文档 + 指定园区文档）。
    """
    parts = [f'tenant_id == "{scope.tenant_id}"']
    if scope.role != "admin":
        parts.append(f'array_contains(role_scope, "{scope.role}")')
    orchards = [o for o in (scope.orchard_scope or []) if o]
    if orchards:
        orchard_checks = " or ".join(
            f'array_contains(orchard_scope, "{o}")' for o in orchards)
        parts.append(f'({orchard_checks} or array_contains(orchard_scope, '
                     f'"{ALL_ORCHARDS}"))')
    return " and ".join(parts)


def scope_from_graph_state(state) -> DataScope:
    """从图状态提取用户数据范围（tenant/role/orchard_scope）。"""
    return DataScope(
        tenant_id=state.get("tenant_id", "default"),
        role=state.get("role", "worker"),
        orchard_scope=state.get("user_orchard_scope")
        or state.get("orchard_scope") or [],
    )
