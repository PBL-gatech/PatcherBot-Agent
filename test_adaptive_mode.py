import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _parse(relative_path):
    source = (ROOT / relative_path).read_text(encoding='utf-8-sig')
    return ast.parse(source)


def _class(module, name):
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_node, name):
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _body_dump(body):
    module = ast.Module(body=body, type_ignores=[])
    return ast.dump(module, include_attributes=False)


def _mode_comparison_value(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], ast.Eq) or len(node.comparators) != 1:
        return None
    left = node.left
    comparator = node.comparators[0]
    if not isinstance(left, ast.Attribute) or left.attr != 'mode':
        return None
    if not isinstance(comparator, ast.Constant):
        return None
    return comparator.value


def _assigned_mode_value(method, variable_name):
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == variable_name:
            return _mode_comparison_value(node.value)
    raise AssertionError(f'No assignment found for {variable_name}')


def _flag_branch_bodies(method, flag_name):
    branches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == flag_name
    ]
    return [_body_dump(node.body) for node in sorted(branches, key=lambda node: node.lineno)]


def test_adaptive_mode_is_selectable_and_classic_remains_default():
    module = _parse('patcherbot/interface/patchConfig.py')
    patch_config = _class(module, 'PatchConfig')
    mode_assignment = next(
        node
        for node in patch_config.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'mode' for target in node.targets)
    )
    selector = mode_assignment.value
    keywords = {keyword.arg: keyword.value for keyword in selector.keywords}

    assert ast.literal_eval(keywords['default']) == 'Classic'
    assert ast.literal_eval(keywords['objects']) == [
        'Manual',
        'Classic',
        'Adaptive',
        'Agent',
        'Training',
    ]


def test_adaptive_logger_code_is_distinct_and_mapped():
    module = _parse('patcherbot/utils/StateMachineLogger.py')
    logger = _class(module, 'StateMachineLogger')
    constants = {}
    for node in logger.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            constants[target.id] = node.value.value

    assert constants['CLASSIC'] == 0
    assert constants['MANUAL'] == 1
    assert constants['AGENT'] == 2
    assert constants['TRAINING'] == 3
    assert constants['NOMODE'] == 4
    assert constants['ADAPTIVE'] == 5

    mode_map = next(
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'mode_map'
    )
    mapping = {
        key.value: value.attr
        for key, value in zip(mode_map.keys, mode_map.values)
    }
    assert mapping['Adaptive'] == 'ADAPTIVE'


def test_adaptive_duplicates_the_non_agent_pipette_alignment_block():
    module = _parse('patcherbot/controller/patch.py')
    find_pipette = _method(_class(module, 'AutoPatcher'), 'find_pipette')
    adaptive = next(
        node
        for node in ast.walk(find_pipette)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == 'adaptive_mode'
    )
    classic_non_agent = next(
        node
        for node in ast.walk(find_pipette)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == 'should_act'
    )

    assert _body_dump(adaptive.body) == _body_dump(classic_non_agent.body)


def test_adaptive_has_an_independent_classic_hunt_copy():
    module = _parse('patcherbot/controller/patch.py')
    hunt_cell = _method(_class(module, 'AutoPatcher'), 'hunt_cell')
    classic = next(
        node
        for node in ast.walk(hunt_cell)
        if isinstance(node, ast.If) and _mode_comparison_value(node.test) == 'Classic'
    )
    adaptive = classic.orelse[0]

    assert isinstance(adaptive, ast.If)
    assert _mode_comparison_value(adaptive.test) == 'Adaptive'
    assert _body_dump(adaptive.body) == _body_dump(classic.body)


def test_adaptive_has_independent_classic_control_copies():
    module = _parse('patcherbot/controller/patch.py')
    auto_patcher = _class(module, 'AutoPatcher')

    for method_name in ('gigaseal', 'break_in'):
        method = _method(auto_patcher, method_name)
        assert _assigned_mode_value(method, 'autoPressure') == 'Classic'
        assert _assigned_mode_value(method, 'adaptivePressure') == 'Adaptive'

        classic_bodies = _flag_branch_bodies(method, 'autoPressure')
        adaptive_bodies = _flag_branch_bodies(method, 'adaptivePressure')
        assert len(classic_bodies) == 2
        assert adaptive_bodies == classic_bodies
