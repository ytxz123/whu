from __future__ import annotations

from pathlib import Path
from typing import Any


def message_by_role(row: dict[str, Any], role: str) -> dict[str, Any] | None:
    wanted = str(role).strip().lower()
    for msg in row.get("messages", []) or []:
        if str(msg.get("role", "")).strip().lower() == wanted:
            return msg
    return None


def clean_user_prompt(text: Any) -> str:
    return str(text or "").replace("<image>", "").strip()


def extract_prompt(row: dict[str, Any], fallback: str) -> str:
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        return clean_user_prompt(row["prompt"])
    msg = message_by_role(row, "user")
    if msg is not None:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)
        prompt = clean_user_prompt(content)
        if prompt:
            return prompt
    return fallback


def extract_assistant_payload(row: dict[str, Any]) -> Any:
    for key in ("target", "json", "lines"):
        if key in row:
            value = row[key]
            return {"lines": value} if key == "lines" and isinstance(value, list) else value
    msg = message_by_role(row, "assistant")
    if msg is not None:
        content = msg.get("content", {"lines": []})
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            return content
    return {"lines": []}


def extract_image_path(row: dict[str, Any], root: Path) -> str:
    value = row.get("image")
    if not value:
        images = row.get("images", []) or []
        if images:
            value = images[0]
    if not value:
        raise KeyError("Dataset row must contain either 'image' or a non-empty 'images' list.")
    path = Path(str(value))
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)
