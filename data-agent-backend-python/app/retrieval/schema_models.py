from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class DbTableSimpleView(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str | UUID] = None
    name: str


class DbColumnSchemaView(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str | UUID] = None
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    is_primary_key: bool = Field(default=False, alias="isPrimaryKey")
    db_table: Optional[DbTableSimpleView] = Field(default=None, alias="dbTable")


class DbTableSchemaView(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str | UUID] = None
    name: str
    description: Optional[str] = None
    columns: list[DbColumnSchemaView] = Field(default_factory=list)


class DbForeignKeySchemaView(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str | UUID] = None
    source_column: DbColumnSchemaView = Field(alias="sourceColumn")
    target_column: DbColumnSchemaView = Field(alias="targetColumn")

    def to_expression(self) -> str:
        s_tbl = self.source_column.db_table.name if self.source_column.db_table else "?"
        t_tbl = self.target_column.db_table.name if self.target_column.db_table else "?"
        return f"{s_tbl}.{self.source_column.name}={t_tbl}.{self.target_column.name}"


class Schema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    database_id: str = Field(alias="databaseId")
    db_tables: list[DbTableSchemaView] = Field(default_factory=list, alias="dbTables")
    db_foreign_keys: list[DbForeignKeySchemaView] = Field(default_factory=list, alias="dbForeignKeys")
    enable_example_sampling: bool = Field(default=False, alias="enableExampleSampling")

    def build_table_prompt(
        self, db_table: DbTableSchemaView, examples_by_column: Optional[dict[str, list[str]]] = None
    ) -> str:
        primary_keys = {col.name for col in db_table.columns if col.is_primary_key}
        lines = [f"# Table: {db_table.name}", "["]
        for col in db_table.columns:
            pk_str = "primaryKey, " if col.name in primary_keys else ""
            examples = (examples_by_column or {}).get(col.name, [])
            ex_str = f"Examples: [{', '.join(examples)}]" if examples else "Examples: []"
            # 标准结构化列定义与示例格式化输出
            lines.append(f"({col.name}: {col.type or 'TEXT'}\n, {col.description or ''}, {pk_str}{ex_str}),")
        lines.append("]")
        return "\n".join(lines) + "\n"

    def build_scheme_prompt(self) -> str:
        sb = [f"【DB_ID】 {self.database_id}\n"]
        for table in self.db_tables:
            sb.append(self.build_table_prompt(table))
        keys = "\n".join(fk.to_expression() for fk in self.db_foreign_keys)
        sb.append(f"【Foreign keys】\n{keys}")
        return "".join(sb)

