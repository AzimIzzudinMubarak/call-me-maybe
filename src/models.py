from typing import Any, Literal
from pydantic import BaseModel


class FunctionParameter(BaseModel):
    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParameter]


class PromptInput(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
