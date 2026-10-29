"""Bounded, offline AST evidence for source-survey component classification.

This module parses text; it never imports or executes surveyed source. Its
classes describe inspected components, not repository completeness, execution,
RLVR reward provenance, implementation independence, or objective correctness.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Iterable


ENGINE = "quotientguard.survey-classify.v2"
_LOG = re.compile(r"log_?(?:prob|p)(?:s)?", re.I)
_ADV = re.compile(r"(?:advantage|(?:^|_)adv(?:$|_))", re.I)
_LOSS = re.compile(r"(?:^|_)(?:loss|losses)$", re.I)
_TRAINER_MODULES = {"trl", "transformers", "verl", "openrlhf", "accelerate", "rlhf"}
_MAX_BYTES = 180_000
_MAX_AST_DEPTH = 160
_MAX_AST_NODES = 60_000


def _walk_scope(node: ast.AST) -> Iterable[ast.AST]:
    """Walk one lexical scope, excluding bodies of child scopes."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from _walk_scope(child)


def _scopes(tree: ast.AST, prefix: str = "", ancestors: tuple = ()) -> Iterable[tuple[str, ast.AST, tuple]]:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{prefix}.{node.name}" if prefix else node.name
            if not isinstance(node, ast.ClassDef):
                yield name, node, ancestors
            # Class namespaces are not enclosing lexical scopes for methods.
            inherited = ancestors if isinstance(node, ast.ClassDef) else (*ancestors, node)
            yield from _scopes(node, name, inherited)
        else:
            yield from _scopes(node, prefix, ancestors)


def _ast_limit(tree: ast.AST) -> str | None:
    """Check resource bounds iteratively before any recursive analysis."""
    pending = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if depth > _MAX_AST_DEPTH:
            return "AST_DEPTH_LIMIT"
        if count > _MAX_AST_NODES:
            return "AST_NODE_LIMIT"
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return None


def _scope_imports(inherited: dict[str, str], scope: ast.AST) -> dict[str, str]:
    """Carry lexical import identities; discard identities shadowed by bindings.

    Binding order/control flow and explicit global/nonlocal declarations are not
    resolved. A possible rebinding therefore removes a confirmed identity.
    """
    imports = dict(inherited)
    nodes = list(_walk_scope(scope))
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
    shadowed = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
    for node in nodes:
        if isinstance(node, ast.ExceptHandler) and node.name:
            shadowed.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            shadowed.update(node.names)
        # _walk_scope excludes child scopes entirely, so inspect their names.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                shadowed.add(child.name)
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs,
                     *([scope.args.vararg] if scope.args.vararg else []), *([scope.args.kwarg] if scope.args.kwarg else [])]
        shadowed.update(argument.arg for argument in arguments)
    for name in shadowed:
        imports.pop(name, None)
    return imports


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _identifiers(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def _annotation(node: ast.AST, source: str, path: str, digest: str) -> dict[str, Any]:
    excerpt = ast.get_source_segment(source, node) or ""
    return {
        "path": path, "source_sha256": digest,
        "start_line": getattr(node, "lineno", 1),
        "end_line": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
        "excerpt": excerpt[:1200], "excerpt_truncated": len(excerpt) > 1200,
    }


def _assignments(nodes: list[ast.AST]) -> dict[str, list[tuple[int, ast.AST, ast.AST]]]:
    assignments: dict[str, list[tuple[int, ast.AST, ast.AST]]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append((node.lineno, value, node))
    for values in assignments.values():
        values.sort(key=lambda value: value[0])
    return assignments


def _resolve(node: ast.AST, assignments: dict, before: int, seen: frozenset = frozenset(), depth: int = 0, budget: list[int] | None = None) -> ast.AST:
    """Expand simple local assignments into a bounded symbolic AST copy."""
    if budget is None:
        budget = [20_000]
    budget[0] -= 1
    if depth > 24 or budget[0] < 0:
        raise ValueError("expression expansion depth limit")
    if isinstance(node, ast.Name) and node.id not in seen:
        choices = [value for value in assignments.get(node.id, []) if value[0] < before]
        if choices:
            line, value, _ = choices[-1]
            return _resolve(value, assignments, line, seen | {node.id}, depth + 1, budget)
    fields: dict[str, Any] = {}
    for key, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            fields[key] = _resolve(value, assignments, before, seen, depth + 1, budget)
        elif isinstance(value, list):
            fields[key] = [_resolve(v, assignments, before, seen, depth + 1, budget) if isinstance(v, ast.AST) else v for v in value]
        else:
            fields[key] = value
    return type(node)(**fields)


def _ratio(node: ast.AST) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if _dotted(call.func).split(".")[-1] != "exp":
            continue
        argument = call.args[0] if call.args else call.func.value if isinstance(call.func, ast.Attribute) else None
        if argument is None:
            continue
        for sub in ast.walk(argument):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Sub):
                if any(_LOG.search(x) for x in _identifiers(sub.left)) and any(_LOG.search(x) for x in _identifiers(sub.right)):
                    return True
    return False


def _policy_product(node: ast.AST) -> bool:
    for product in ast.walk(node):
        if isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult):
            for ratio, advantage in ((product.left, product.right), (product.right, product.left)):
                if _ratio(ratio) and any(_ADV.search(x) for x in _identifiers(advantage)):
                    return True
    return False


def _straight_line(scope: ast.AST) -> bool:
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if not scope.body or not isinstance(scope.body[-1], ast.Return) or sum(isinstance(n, ast.Return) for n in scope.body) != 1:
        return False
    for statement in scope.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return)):
            return False
        if isinstance(statement, ast.Assign) and any(not isinstance(t, ast.Name) for t in statement.targets):
            return False
        if any(isinstance(n, (ast.IfExp, ast.NamedExpr, ast.Await, ast.Yield, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(statement)):
            return False
    return True


def _dependency_statements(sink: ast.AST, assignments: dict) -> list[ast.AST]:
    used = {id(sink): sink}
    pending = [(sink.value, sink.lineno)]
    while pending and len(used) <= 2000:
        expression, before = pending.pop()
        for node in ast.walk(expression):
            if not isinstance(node, ast.Name):
                continue
            choices = [v for v in assignments.get(node.id, []) if v[0] < before]
            if choices:
                line, value, statement = choices[-1]
                if id(statement) not in used:
                    used[id(statement)] = statement
                    pending.append((value, line))
    return sorted(used.values(), key=lambda node: node.lineno)


def _resolved_call(node: ast.AST, imports: dict[str, str]) -> str:
    dotted = _dotted(node)
    root, _, rest = dotted.partition(".")
    return imports.get(root, root) + (f".{rest}" if rest else "")


def classify_python(source: str, path: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Classify one file, containing recursion/resource failures for batch use."""
    try:
        return _classify_python(source, path, expected_sha256=expected_sha256)
    except RecursionError:
        return {"engine": ENGINE, "path": path, "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "status": "ANALYSIS_LIMIT", "components": [], "analysis_limit": "ANALYZER_RECURSION_LIMIT",
                "limitations": ["Analysis could not complete within recursion bounds; no negative source/repository conclusion is available."]}


def _classify_python(source: str, path: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Return source-bound static component evidence, with conservative unknowns.

    A local objective requires a returned/backward-connected product of an
    exponentiated log-probability difference and an advantage expression. This
    intentionally misses other valid estimators. Identifiers identify candidates;
    AST connectivity prevents disconnected keywords from establishing a loss.
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    result: dict[str, Any] = {
        "engine": ENGINE, "path": path, "source_sha256": digest,
        "status": "INSPECTED", "components": [],
        "limitations": ["Static source component evidence only; execution, tensor semantics and reward/verifier provenance unknown.",
                        "No negative repository-level inference from missing recognized expressions."],
    }
    if expected_sha256 is not None and expected_sha256 != digest:
        result["status"] = "HASH_MISMATCH"
        return result
    if len(source.encode("utf-8")) > _MAX_BYTES:
        result["status"] = "SIZE_LIMIT"
        return result
    # Python's byte-source parser consumes one leading UTF-8 BOM. Text callers
    # retain it after utf-8 decoding. Normalize parsing and excerpt offsets,
    # while hashes and size limits above attest the original source bytes.
    source = source.removeprefix("\ufeff")
    try:
        tree = ast.parse(source)
    except RecursionError:
        result.update(status="ANALYSIS_LIMIT", analysis_limit="AST_PARSER_RECURSION_LIMIT")
        return result
    except (SyntaxError, ValueError):
        result["status"] = "SYNTAX_ERROR"
        return result
    limit = _ast_limit(tree)
    if limit:
        result.update(status="ANALYSIS_LIMIT", analysis_limit=limit)
        return result
    imports = _scope_imports({}, tree)
    test_path = any(part in {"test", "tests"} for part in PurePosixPath(path.replace("\\", "/")).parts) or PurePosixPath(path).name.startswith("test_")
    for symbol, scope, ancestors in [("<module>", tree, ()), *_scopes(tree)]:
        nodes = list(_walk_scope(scope))
        local_imports = dict(imports)
        for enclosing in ancestors:
            local_imports = _scope_imports(local_imports, enclosing)
        local_imports = _scope_imports(local_imports, scope)
        assignments = _assignments(nodes)
        sinks: list[tuple[ast.AST, ast.AST]] = []
        for node in nodes:
            if isinstance(node, ast.Return) and node.value is not None:
                sinks.append((node, node.value))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "backward":
                sinks.append((node, node.func.value))
        connected: list[tuple[ast.AST, ast.AST]] = []
        for sink, value in sinks:
            loss_context = (not isinstance(sink, ast.Return) or "loss" in symbol.lower()
                            or any(_LOSS.search(name) for name in _identifiers(value)))
            if not loss_context:
                continue
            try:
                expression = _resolve(value, assignments, sink.lineno + (1 if isinstance(value, ast.Name) else 0))
            except (ValueError, RecursionError):
                continue
            if _policy_product(expression):
                connected.append((sink, expression))
        if connected:
            straight = _straight_line(scope) and len(connected) == 1 and isinstance(connected[0][0], ast.Return)
            sink, expression = connected[0]
            contract: dict[str, Any] = {
                "status": "STATIC_EXPRESSION_GRAPH_ONLY" if straight else "UNKNOWN",
                "full_objective_support": False,
                "unknowns": ["Tensor shapes, masks and auxiliary terms require declared contracts.",
                             "Dynamic dispatch, optimizer path, runtime values and RLVR verifier provenance unverified."],
            }
            if straight:
                graph = ast.dump(expression, include_attributes=False)
                contract.update({"expression": ast.unparse(expression),
                                 "expression_ast_sha256": hashlib.sha256(graph.encode()).hexdigest(),
                                 "source_statements": [_annotation(n, source, path, digest) for n in _dependency_statements(sink, assignments)]})
            else:
                contract["reason"] = "Control flow, nested execution or multiple sinks prevent unambiguous whole-function reconstruction."
            result["components"].append({
                **_annotation(scope if symbol != "<module>" else sink, source, path, digest),
                "symbol": symbol,
                "kind": "TEST_POLICY_OBJECTIVE_EXAMPLE" if test_path else "LOCAL_POLICY_OBJECTIVE_IMPLEMENTATION",
                "status": "STATIC_COMPONENT_CONFIRMED" if straight else "STATIC_COMPONENT_CANDIDATE",
                "evidence_basis": "Loss-named return/backward AST expression connects exp(log-probability subtraction) to advantage multiplication.",
                "sink": _annotation(sink, source, path, digest), "contract": contract,
                "independence_status": "UNKNOWN", "execution_status": "NOT_EXECUTED",
            })
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "train" and isinstance(node.func.value, ast.Name):
                preceding = [value for value in assignments.get(node.func.value.id, []) if value[0] < node.lineno]
                binding = preceding[-1] if preceding else None
                call = binding[1] if binding and isinstance(binding[1], ast.Call) else None
                imported = call is not None and _dotted(call.func).split(".")[0] in local_imports
                name = _resolved_call(call.func, local_imports) if imported else ""
                if binding and name.split(".")[0] in _TRAINER_MODULES and name.split(".")[-1].endswith("Trainer"):
                    result["components"].append({**_annotation(node, source, path, digest), "symbol": symbol,
                        "kind": "EXTERNAL_TRAINER_WRAPPER", "status": "STATIC_COMPONENT_CONFIRMED",
                        "external_constructor": name, "constructor": _annotation(binding[2], source, path, digest),
                        "evidence_basis": "Imported external Trainer construction followed by train call; external objective not inspected."})
        data_calls = [n for n in nodes if isinstance(n, ast.Call) and (
            _resolved_call(n.func, local_imports) in {"datasets.load_dataset", "datasets.load_from_disk"}
            or _dotted(n.func).split(".")[-1] == "evaluate")]
        if data_calls:
            result["components"].append({**_annotation(data_calls[0], source, path, digest), "symbol": symbol,
                "kind": "EVALUATION_OR_DATA_COMPONENT_CANDIDATE", "status": "CANDIDATE",
                "evidence_basis": "Dataset loading or evaluation call observed; may also be used during training."})
    return result


def compare_objective_components(base: dict, changed: dict) -> list[dict[str, Any]]:
    """Compare matching-path/symbol static graphs, never infer ancestry from SHA."""
    if base["path"] != changed["path"]:
        return []
    base_components = {c["symbol"]: c for c in base["components"] if c.get("contract", {}).get("expression_ast_sha256")}
    comparisons = []
    for component in changed["components"]:
        previous = base_components.get(component["symbol"])
        signature = component.get("contract", {}).get("expression_ast_sha256")
        if previous is None or signature is None:
            continue
        comparisons.append({"path": changed["path"], "symbol": component["symbol"],
            "base_source_sha256": base["source_sha256"], "changed_source_sha256": changed["source_sha256"],
            "base_expression_ast_sha256": previous["contract"]["expression_ast_sha256"],
            "changed_expression_ast_sha256": signature,
            "status": "SAME_STATIC_OBJECTIVE_EXPRESSION" if signature == previous["contract"]["expression_ast_sha256"] else "STATIC_OBJECTIVE_EXPRESSION_CHANGED",
            "semantic_equivalence_status": "NOT_PROVED", "execution_status": "NOT_EXECUTED"})
    return comparisons


def classify_repository(files: list[dict], *, omitted_paths: list[str] | None = None, relationship: dict | None = None) -> dict[str, Any]:
    """Summarize inspected components, explicitly retaining repository unknowns."""
    omitted_paths = omitted_paths or []
    relationship = relationship or {}
    components = [c for f in files for c in f.get("components", [])]
    counts = Counter(c["kind"] for c in components)
    if counts["LOCAL_POLICY_OBJECTIVE_IMPLEMENTATION"]:
        selected = "LOCAL_POLICY_IMPLEMENTATION_INDEPENDENCE_UNKNOWN"
    elif counts["EXTERNAL_TRAINER_WRAPPER"]:
        selected = "EXTERNAL_TRAINER_WRAPPER"
    elif counts["EVALUATION_OR_DATA_COMPONENT_CANDIDATE"]:
        selected = "EVALUATION_OR_DATA_COMPONENT_CANDIDATE"
    elif counts["TEST_POLICY_OBJECTIVE_EXAMPLE"]:
        selected = "TEST_EXAMPLE_ONLY_IN_INSPECTED_SCOPE"
    else:
        selected = "UNKNOWN"
    return {"whole_repository_classification": "UNKNOWN", "selected_component_classification": selected,
        "classification_scope": "INSPECTED_PYTHON_COMPONENTS_ONLY", "component_counts": dict(sorted(counts.items())),
        "inspected_paths": [{"path": f["path"], "source_sha256": f.get("source_sha256"), "status": f["status"]} for f in files],
        "omitted_paths": omitted_paths, "omitted_path_count": len(omitted_paths),
        "omission_reason": "Prior bounded survey path selection; no claim about uninspected or non-Python code.",
        "independent_implementation_confirmed": False,
        "relationship_status": "FORK_METADATA_ONLY" if relationship.get("is_fork") else "UNKNOWN",
        "relationship": relationship,
        "unknown_reasons": ["Fork flags, distinct source hashes and source-pattern deduplication cannot establish independent authorship.",
                            "Whole-repository purpose and exclusive eval/data/irrelevant status are not established by partial Python coverage.",
                            "Recognized policy expression does not establish verifiable rewards or actual training execution."]}
