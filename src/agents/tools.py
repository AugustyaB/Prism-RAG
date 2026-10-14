"""Curated Agent Tools, Safe AST Calculator & Function Calling Registry (L2)."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from loguru import logger
from qdrant_client.http import models as qmodels
from sqlalchemy import select

from src.database.models import DocumentRecord
from src.database.relational import SessionLocal

# ── Safe AST Arithmetic Evaluator ──────────────────────────────────────────

SAFE_OPERATORS: dict[type[ast.AST], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS: dict[str, Callable] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
}


def _safe_eval_node(node: ast.AST) -> int | float:
    """Recursively evaluate a validated Python AST arithmetic expression node."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](_safe_eval_node(node.operand))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            func_name = getattr(node.func, "id", str(node.func))
            raise ValueError(f"Disallowed function call '{func_name}'. Only safe math functions allowed.")
        args = [_safe_eval_node(arg) for arg in node.args]
        return SAFE_FUNCTIONS[node.func.id](*args)
    else:
        raise ValueError(f"Disallowed AST expression syntax: {type(node).__name__}")


def deterministic_calculator(expression: str) -> dict[str, Any]:
    """Safely evaluate an arithmetic expression using pure Python AST (zero eval/exec)."""
    clean_expr = expression.strip()
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_node(parsed)
        return {
            "tool": "deterministic_calculator",
            "expression": clean_expr,
            "result": result,
            "status": "success",
        }
    except Exception as exc:
        logger.warning("Deterministic calculator rejected expression '{}': {}", clean_expr, exc)
        return {
            "tool": "deterministic_calculator",
            "expression": clean_expr,
            "error": str(exc),
            "status": "error",
        }


# ── Document Tools ─────────────────────────────────────────────────────────


def document_search(
    query: str,
    retriever: Any = None,
    mode: str = "hybrid_rrf",
    user: Any = None,
) -> dict[str, Any]:
    """Execute dynamic hybrid search against the vector store."""
    if retriever is None:
        return {"tool": "document_search", "error": "Retriever not configured", "status": "error"}

    try:
        outcome = retriever.retrieve(
            query=query,
            mode=mode,
            user=user,
            enable_multi_query=False,
            enable_crag=False,
        )
        return {
            "tool": "document_search",
            "query": query,
            "top_score": outcome.get("top_score"),
            "abstained": outcome.get("abstained", False),
            "results_count": len(outcome.get("results", [])),
            "context_excerpt": outcome.get("context", "")[:600],
            "status": "success",
        }
    except Exception as exc:
        return {"tool": "document_search", "error": str(exc), "status": "error"}


def document_summary(filename: str, vector_store: Any = None) -> dict[str, Any]:
    """Extract cover page, abstract, and TOC metadata for a specified document."""
    doc_info: dict[str, Any] = {"filename": filename, "tool": "document_summary"}

    # Query relational database for document metadata
    with SessionLocal() as db:
        stmt = select(DocumentRecord).where(DocumentRecord.filename.like(f"%{filename}%"))
        doc = db.scalars(stmt).first()
        if doc:
            doc_info.update(
                {
                    "total_pages": doc.total_pages,
                    "total_chunks": doc.total_chunks,
                    "visibility": doc.visibility,
                    "file_hash": doc.file_hash,
                }
            )

    # Search for first 2 pages in vector store if available
    summary_snippets: list[str] = []
    if vector_store is not None and hasattr(vector_store, "client"):
        try:
            q_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_pdf",
                        match=qmodels.MatchValue(value=filename),
                    )
                ]
            )
            hits = vector_store.search(
                query="abstract introduction overview table of contents",
                initial_k=10,
                final_k=3,
                mode="hybrid_rrf",
                query_filter=q_filter,
            )
            for h in hits:
                txt = h.get("parent_text") or h.get("text", "")
                if txt and txt not in summary_snippets:
                    summary_snippets.append(txt[:300].strip())
        except Exception as exc:
            logger.debug("Vector store lookup failed in document_summary: {}", exc)

    doc_info["summary_excerpt"] = "\n\n---\n\n".join(summary_snippets) if summary_snippets else "No abstract content indexed."
    doc_info["status"] = "success"
    return doc_info


def document_comparison(
    doc_a: str,
    doc_b: str,
    topic: str,
    retriever: Any = None,
    vector_store: Any = None,
) -> dict[str, Any]:
    """Execute dual-retrieval comparative matrix between two documents on a specific topic."""
    result_a: list[str] = []
    result_b: list[str] = []

    if vector_store is not None:
        try:
            filter_a = qmodels.Filter(
                must=[qmodels.FieldCondition(key="source_pdf", match=qmodels.MatchValue(value=doc_a))]
            )
            hits_a = vector_store.search(query=topic, initial_k=5, final_k=2, mode="hybrid_rrf", query_filter=filter_a)
            result_a = [h.get("text", "")[:250].strip() for h in hits_a]

            filter_b = qmodels.Filter(
                must=[qmodels.FieldCondition(key="source_pdf", match=qmodels.MatchValue(value=doc_b))]
            )
            hits_b = vector_store.search(query=topic, initial_k=5, final_k=2, mode="hybrid_rrf", query_filter=filter_b)
            result_b = [h.get("text", "")[:250].strip() for h in hits_b]
        except Exception as exc:
            logger.warning("Document comparison retrieval failed: {}", exc)

    return {
        "tool": "document_comparison",
        "doc_a": doc_a,
        "doc_b": doc_b,
        "topic": topic,
        "doc_a_findings": result_a,
        "doc_b_findings": result_b,
        "status": "success",
    }


# ── Tool Registry & JSON Schema Definitions ────────────────────────────────

TOOL_REGISTRY: dict[str, Callable] = {
    "deterministic_calculator": deterministic_calculator,
    "document_search": document_search,
    "document_summary": document_summary,
    "document_comparison": document_comparison,
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return standard JSON function schemas for LLM tool calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "deterministic_calculator",
                "description": "Safely evaluates mathematical and arithmetic expressions without code execution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g. '(145 * 12) / 4'.",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "document_search",
                "description": "Performs hybrid retrieval against the document knowledge base for a specific sub-topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Focused search query to retrieve relevant chunks."}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "document_summary",
                "description": "Extracts cover page, abstract, and structural metadata for an ingested document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Filename of the document to summarize."}
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "document_comparison",
                "description": "Performs side-by-side comparative retrieval between two documents on a specified topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_a": {"type": "string", "description": "First document filename."},
                        "doc_b": {"type": "string", "description": "Second document filename."},
                        "topic": {"type": "string", "description": "The specific topic or dimension to compare."},
                    },
                    "required": ["doc_a", "doc_b", "topic"],
                },
            },
        },
    ]


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    retriever: Any = None,
    vector_store: Any = None,
    user: Any = None,
) -> dict[str, Any]:
    """Dispatch a tool call to the corresponding handler in TOOL_REGISTRY."""
    if tool_name not in TOOL_REGISTRY:
        return {"tool": tool_name, "error": f"Tool '{tool_name}' not registered.", "status": "error"}

    handler = TOOL_REGISTRY[tool_name]
    logger.info("Executing tool '{}' with arguments: {}", tool_name, arguments)

    try:
        if tool_name == "deterministic_calculator":
            return handler(arguments.get("expression", ""))
        elif tool_name == "document_search":
            return handler(arguments.get("query", ""), retriever=retriever, user=user)
        elif tool_name == "document_summary":
            return handler(arguments.get("filename", ""), vector_store=vector_store)
        elif tool_name == "document_comparison":
            return handler(
                doc_a=arguments.get("doc_a", ""),
                doc_b=arguments.get("doc_b", ""),
                topic=arguments.get("topic", ""),
                retriever=retriever,
                vector_store=vector_store,
            )
        else:
            return handler(**arguments)
    except Exception as exc:
        logger.error("Error executing tool '{}': {}", tool_name, exc)
        return {"tool": tool_name, "error": str(exc), "status": "error"}
