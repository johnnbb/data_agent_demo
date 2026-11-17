import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.retrieval.schema_models import Schema

logger = logging.getLogger(__name__)

DEFAULT_ENUM_MAPPING_PATH = Path(__file__).resolve().parents[2] / "data" / "enum_mapping.json"


@lru_cache(maxsize=4)
def load_enum_mappings(mapping_path: Optional[str] = None) -> dict[str, Any]:
    """加载枚举值语义映射字典（按 database_id -> table -> column 组织）。"""
    path = Path(mapping_path) if mapping_path else DEFAULT_ENUM_MAPPING_PATH
    if not path.exists():
        logger.warning(f"[EnumRules] Mapping file not found: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"[EnumRules] Failed to load enum mappings from {path}: {exc}")
        return {}


def format_semantic_enum_rules(
    database_id: str,
    schema: Optional[Schema] = None,
    mapping_path: Optional[str] = None,
) -> str:
    """
    根据当前 database_id 与已召回的局部 Schema（表与列），
    过滤并格式化命中的枚举值语义映射规则，避免注入无关表噪音。
    未命中任何规则时返回 "(无)"。
    """
    if not database_id:
        return "(无)"

    all_mappings = load_enum_mappings(mapping_path)
    db_mappings = all_mappings.get(database_id)
    if not isinstance(db_mappings, dict) or not db_mappings:
        return "(无)"

    active_cols_by_table: Optional[dict[str, set[str]]] = None
    if schema is not None and schema.db_tables:
        active_cols_by_table = {
            t.name: {c.name for c in t.columns} if t.columns else set()
            for t in schema.db_tables
        }

    lines: list[str] = []
    for table_name, cols_dict in db_mappings.items():
        if active_cols_by_table is not None and table_name not in active_cols_by_table:
            continue
        if not isinstance(cols_dict, dict):
            continue

        allowed_cols = active_cols_by_table.get(table_name) if active_cols_by_table is not None else None
        for col_name, spec in cols_dict.items():
            # 若表存在列集合且当前列不在列集合中则跳过（允许空列集合兜底匹配整表）
            if allowed_cols and col_name not in allowed_cols:
                continue
            if not isinstance(spec, dict):
                continue
            mappings = spec.get("mappings")
            if not isinstance(mappings, dict) or not mappings:
                continue
            desc = spec.get("description") or ""
            pairs = "; ".join(f"'{code}' = {meaning}" for code, meaning in mappings.items())
            header = f"- {table_name}.{col_name}"
            if desc:
                header += f" ({desc})"
            lines.append(f"{header}: {pairs}")

    return "\n".join(lines) if lines else "(无)"
