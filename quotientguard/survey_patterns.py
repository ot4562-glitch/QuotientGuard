"""Bounded intraprocedural syntax/dataflow observations, never defect verdicts.

Only assignments, expression provenance and conservative branch joins are
modeled. Dynamic calls, tensor shapes/dtypes, input reachability and relations
between different masks remain unknown. No inspected source is executed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace


POLICY_VERSION = 'quotientguard.patterns.v3'
MAX_NODES = 100_000


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + '.' + node.attr
    return ''


def _mask_name(name):
    return 'mask' in name.lower() or name.lower() in {'m', 'valid', 'accepted', 'rejected'}


@dataclass(frozen=True)
class Value:
    uncertain: bool = True
    masks: frozenset[str] = frozenset()
    guards: frozenset[str] = frozenset()
    operations: frozenset[str] = frozenset()
    detached: bool = False
    mask_count: bool = False
    zero_guarded: bool = False
    refs: frozenset[str] = frozenset()
    literal_zero: bool = False
    outside_zero: bool = False


def _join(a, b, operation=None):
    guards = (a.guards if not b.uncertain else b.guards if not a.uncertain
              else a.guards & b.guards)
    outside_zero = False
    if operation:
        guards = frozenset()
        if operation == 'subtract':
            if a.outside_zero and b.outside_zero:
                guards = a.guards & b.guards
            elif b.literal_zero:
                guards = a.guards if a.outside_zero else frozenset()
            elif a.literal_zero:
                guards = b.guards if b.outside_zero else frozenset()
        elif operation == 'multiply':
            guards = (a.masks & b.guards) | (b.masks & a.guards)
        outside_zero = bool(guards)
    return Value(uncertain=a.uncertain or b.uncertain, guards=guards, outside_zero=outside_zero,
        operations=a.operations | b.operations | ({operation} if operation else set()),
        # Literal/mask operands cannot reconnect a detached value to autograd.
        detached=(a.detached or not a.uncertain) and (b.detached or not b.uncertain)
                 and (a.detached or b.detached),
        refs=frozenset(sorted(a.refs | b.refs)[:16]))


def _branch_join(a, b):
    if a == b:
        return a
    return replace(_join(a, b), masks=a.masks & b.masks,
                   mask_count=a.mask_count and b.mask_count,
                   zero_guarded=a.zero_guarded and b.zero_guarded)


def _inverse(masks):
    return frozenset(x[1:] if x.startswith('!') else '!' + x for x in masks)


def _number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        number = _number(node.operand)
        return -number if number is not None else None
    return None


class Analyzer:
    def __init__(self):
        self.candidates = {}
        self.observations = {}

    def candidate(self, node, pattern, priority, note, value):
        key = (node.lineno, node.col_offset, pattern)
        self.candidates[key] = {'_node': node, 'line': node.lineno, 'column': node.col_offset,
            'pattern': pattern, 'priority': priority, 'reason': note,
            'classification': 'STATIC_RISK_CANDIDATE',
            'operations': sorted(value.operations), 'dependencies': sorted(value.refs),
            'input_reachability': 'UNKNOWN', 'objective_contract': 'UNKNOWN'}

    def observe(self, node, kind, **details):
        self.observations[(node.lineno, node.col_offset, kind)] = {
            'line': node.lineno, 'kind': kind, **details}

    def expr(self, node, env):
        if node is None:
            return Value(uncertain=False)
        if isinstance(node, ast.Constant):
            return Value(uncertain=not isinstance(node.value, (int, float, bool, type(None))),
                         literal_zero=isinstance(node.value, (int, float, bool)) and node.value == 0)
        if isinstance(node, ast.Name):
            return env.get(node.id, Value(masks=frozenset({node.id}) if _mask_name(node.id) else frozenset(),
                uncertain=not _mask_name(node.id), refs=frozenset({node.id})))
        if isinstance(node, ast.Attribute):
            name = _name(node)
            if _mask_name(node.attr):
                return Value(uncertain=False, masks=frozenset({name}), refs=frozenset({name}))
            return self.expr(node.value, env)
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value, env)
            self.expr(node.slice, env)
            # Indexing changes alignment; do not carry a mask-selection proof.
            return replace(value, guards=frozenset(), masks=frozenset())
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return replace(value, uncertain=False, masks=_inverse(value.masks))
            return value
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            children = ([node.left, *node.comparators] if isinstance(node, ast.Compare) else node.values)
            values = [self.expr(child, env) for child in children]
            refs = frozenset().union(*(v.refs for v in values))
            if (isinstance(node, ast.Compare) and len(node.ops) == 1 and len(values) == 2
                    and values[0].masks and values[1].literal_zero):
                if isinstance(node.ops[0], ast.Eq):
                    return Value(uncertain=False, masks=_inverse(values[0].masks), refs=refs)
                if isinstance(node.ops[0], ast.NotEq):
                    return Value(uncertain=False, masks=values[0].masks, refs=refs)
            return Value(uncertain=False, masks=frozenset({ast.dump(node, include_attributes=False)}), refs=refs)
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            operation = {ast.Sub: 'subtract', ast.Mult: 'multiply', ast.Div: 'divide', ast.Add: 'add'}.get(type(node.op), 'other_arithmetic')
            value = _join(left, right, operation)
            if isinstance(node.op, ast.Mult):
                for coefficient, operand in ((left, right), (right, left)):
                    if (coefficient.detached and coefficient.uncertain
                            and coefficient.operations & {'exp', 'divide', 'sqrt', 'log'}
                            and operand.uncertain and not operand.detached and not operand.masks):
                        self.candidate(node, 'DETACHED_COEFFICIENT_PRODUCT', 75,
                            'A detached arithmetic coefficient multiplies a potentially differentiable value; '
                            'detach does not sanitize NaN coefficients. Verify forward and backward separately; '
                            'coefficient finiteness, requires_grad and input reachability remain unknown.', coefficient)
                for mask, other in ((left, right), (right, left)):
                    if not mask.masks or not other.uncertain or mask.masks & other.guards:
                        continue
                    self.candidate(node, 'MASK_MULTIPLY_NONFINITE', 75,
                        'Excluded multiplication can preserve NaN or form inf*0; operand reachability is unverified.', other)
                    if 'subtract' in other.operations:
                        self.candidate(node, 'MASKED_SUBTRACTION', 85,
                            'Subtraction precedes exclusion; test -inf-(-inf) and exact excluded gradients.', other)
                    if 'exp' in other.operations:
                        self.candidate(node, 'EXP_BEFORE_MASK', 90,
                            'Exponentiation reaches a later multiplicative mask; test overflow and backward.', other)
            if isinstance(node.op, ast.Div) and right.mask_count:
                self.observe(node, 'ACTIVE_TOKEN_DENOMINATOR', zero_guarded=right.zero_guarded,
                    contract='Active count observed; axes, batch scaling and empty policy require verification.')
                if not right.zero_guarded:
                    self.candidate(node, 'MASK_SUM_DENOMINATOR', 90,
                        'Active count has no recognized positive lower bound; empty-mask behavior is unknown.', right)
            return value
        if isinstance(node, ast.IfExp):
            self.expr(node.test, env)
            return _join(self.expr(node.body, env), self.expr(node.orelse, env))
        if isinstance(node, ast.Call):
            name = _name(node.func).split('.')[-1]
            receiver = self.expr(node.func.value, env) if isinstance(node.func, ast.Attribute) else Value(uncertain=False)
            args = [self.expr(arg, env) for arg in node.args]
            keywords = {keyword.arg: self.expr(keyword.value, env) for keyword in node.keywords}
            value = receiver
            for arg in args:
                value = _join(value, arg)
            if name in {'where', 'masked_fill'}:
                if name == 'where':
                    resolved = [args[i] if len(args) > i else keywords.get(key)
                                for i, key in enumerate(('condition', 'input', 'other'))]
                    if any(item is None for item in resolved):
                        return replace(value, guards=frozenset())
                    condition, chosen, alternative = resolved
                elif name == 'masked_fill' and len(args) >= 2:
                    condition, chosen, alternative = args[0], receiver, args[1]
                    condition = replace(condition, masks=_inverse(condition.masks))
                else:
                    return replace(value, guards=frozenset())
                if (chosen.operations & {'exp', 'divide', 'sqrt', 'log'} and not chosen.detached
                        and not condition.masks & chosen.guards):
                    self.candidate(node, 'WHERE_AFTER_UNSAFE_ARITHMETIC', 90,
                        'Selection occurs after arithmetic with potentially nonfinite backward; where alone is not proof of safety.', chosen)
                self.observe(node, 'MASK_SELECTION', selected_after_operations=sorted(chosen.operations),
                    detached=chosen.detached, mask_relation='SYNTAX_ONLY')
                return replace(_join(chosen, alternative),
                    guards=condition.masks if alternative.literal_zero else frozenset(),
                    outside_zero=alternative.literal_zero, masks=frozenset())
            if name in {'exp', 'sqrt', 'log'}:
                operand = args[0] if args else keywords.get('input', receiver)
                return replace(operand, uncertain=True, operations=operand.operations | {name}, masks=frozenset(),
                    guards=operand.guards if operand.outside_zero and name == 'exp' else frozenset(), outside_zero=False)
            if name in {'detach', 'detach_'}:
                operand = args[0] if args else receiver
                self.observe(node, 'STOP_GRADIENT', operations=sorted(operand.operations), dependencies=sorted(operand.refs))
                return replace(operand, detached=True)
            if name in {'sum', 'mean'}:
                operand = args[0] if args and not isinstance(node.func, ast.Attribute) else receiver
                if name == 'mean':
                    self.observe(node, 'FULL_TENSOR_MEAN',
                        contract='Mean uses the selected tensor extent; active-token equivalence is not inferred.')
                if node.args or any(k.arg in {'dim', 'axis'} for k in node.keywords):
                    self.observe(node, 'SEQUENCE_AGGREGATION', operation=name,
                        axes='Explicit axis syntax only; tensor rank remains unknown.')
                return replace(operand, masks=frozenset(), mask_count=bool(operand.masks),
                    zero_guarded=False, operations=operand.operations | {'aggregate_' + name})
            if name in {'clamp', 'clamp_min', 'clip', 'minimum', 'maximum', 'min', 'max'}:
                self.observe(node, 'CLIPPING_OR_REJECTION', operation=name,
                    contract='Clipping/rejection is not a defect; bounds and derivative branch require a contract.')
                positive_bound = False
                bound_nodes = ([node.args[0]] if name == 'clamp_min' and node.args else [])
                bound_nodes += [k.value for k in node.keywords if k.arg == 'min']
                if name in {'max', 'maximum'}:
                    bound_nodes += list(node.args)
                for bound in bound_nodes:
                    if isinstance(bound, ast.Constant) and isinstance(bound.value, (int, float)) and bound.value > 0:
                        positive_bound = True
                counted = receiver.mask_count or any(a.mask_count for a in args)
                # A lower clamp can turn a selected zero into an overflowing
                # exp input. Preserve a zero premise only for known bounds.
                keep_zero = receiver.outside_zero and bool(receiver.guards) and name in {'clamp', 'clamp_min', 'clip'}
                bounds = []
                if name in {'clamp', 'clip'}:
                    bounds.extend(('min' if i == 0 else 'max', n) for i, n in enumerate(node.args[:2]))
                elif name == 'clamp_min' and node.args:
                    bounds.append(('min', node.args[0]))
                bounds.extend((k.arg, k.value) for k in node.keywords if k.arg in {'min', 'max'})
                for kind, bound in bounds:
                    number = _number(bound)
                    if number is None or (kind == 'min' and number > 0) or (kind == 'max' and number < 0):
                        keep_zero = False
                return replace(value, mask_count=counted,
                    zero_guarded=counted and positive_bound,
                    guards=receiver.guards if keep_zero else frozenset(), outside_zero=keep_zero)
            if name == 'nan_to_num':
                operand = args[0] if args else keywords.get('input', receiver)
                self.observe(node, 'RAW_SANITIZED_VALUES', raw_operations=sorted(operand.operations),
                    raw_dependencies=sorted(operand.refs), raw_values='NOT_EXECUTED')
                if operand.uncertain:
                    self.candidate(node, 'NONFINITE_REPLACEMENT', 50,
                        'Retain pre-sanitization tensors; finite sanitized output does not prove a finite original graph.', operand)
                return replace(operand, guards=frozenset())
            if name in {'bool', 'float', 'to', 'clone', 'contiguous'}:
                return receiver
            if name in {'zeros', 'ones', 'zeros_like', 'ones_like'}:
                return Value(uncertain=False, literal_zero=name in {'zeros', 'zeros_like'})
            # A dynamic call may change values, shapes and masks. Never infer safety.
            return replace(value, uncertain=True, guards=frozenset(), masks=frozenset())
        values = [self.expr(child, env) for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)]
        result = Value(uncertain=False)
        for value in values:
            result = _join(result, value)
        return result

    def body(self, statements, env):
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                value = self.expr(statement.value, env)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = (_join(env.get(target.id, Value()), value)
                                         if isinstance(statement, ast.AugAssign) else value)
                    else:
                        self.expr(target, env)
            elif isinstance(statement, ast.If):
                self.expr(statement.test, env)
                before, after = dict(env), dict(env)
                self.body(statement.body, before)
                self.body(statement.orelse, after)
                for key in before.keys() | after.keys():
                    env[key] = _branch_join(before.get(key, env.get(key, Value())), after.get(key, env.get(key, Value())))
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
                # Loops/exception paths are only a conservative single-pass join.
                branch = dict(env)
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self.expr(child, branch)
                self.body(getattr(statement, 'body', []), branch)
                self.body(getattr(statement, 'orelse', []), branch)
                self.body(getattr(statement, 'finalbody', []), branch)
                for handler in getattr(statement, 'handlers', []):
                    self.body(handler.body, branch)
                for key in branch:
                    env[key] = _branch_join(env.get(key, Value()), branch[key])
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self.expr(child, env)


def analyze_tree(tree):
    nodes = []
    for node in ast.walk(tree):
        nodes.append(node)
        if len(nodes) > MAX_NODES:
            raise ValueError('AST node limit exceeded')
    analyzer = Analyzer()
    analyzer.body(tree.body, {})
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            env = {arg.arg: Value(uncertain=not _mask_name(arg.arg),
                masks=frozenset({arg.arg}) if _mask_name(arg.arg) else frozenset(), refs=frozenset({arg.arg}))
                for arg in arguments}
            analyzer.body(node.body, env)
    return analyzer


def iter_patterns(tree):
    for _, candidate in sorted(analyze_tree(tree).candidates.items()):
        yield candidate['_node'], candidate['pattern'], candidate['priority'], candidate['reason']


def analyze_source(source):
    analyzer = analyze_tree(ast.parse(source))
    return {'policy_version': POLICY_VERSION,
        'candidates': [{k: v for k, v in item.items() if k != '_node'}
                       for _, item in sorted(analyzer.candidates.items())],
        'observations': [value for _, value in sorted(analyzer.observations.items())],
        'semantic_confirmation_eligible': False,
        'limitations': ['Bounded intra-function provenance; no interprocedural or tensor-type proof.',
            'Mask names are syntax hints. Different masks, aliases through containers and dynamic calls remain unknown.',
            'No input reachability, defect, clipping error or numerical safety is established.']}
