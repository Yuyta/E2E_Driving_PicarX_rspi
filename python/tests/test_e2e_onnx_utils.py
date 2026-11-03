import onnx
from onnx import TensorProto, helper

from src.e2e.training.onnx_utils import STACK_TRACE_KEY, strip_export_debug_info

SECRET_PATH = b"C:\\Users\\someone\\project\\train.py"


def _node_with_trace(op, inputs, outputs, **attrs):
    node = helper.make_node(op, inputs, outputs, **attrs)
    prop = node.metadata_props.add()
    prop.key = STACK_TRACE_KEY
    prop.value = SECRET_PATH.decode()
    other = node.metadata_props.add()
    other.key = "keep.me"
    other.value = "value"
    return node


def _model(nodes):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(nodes, "g", [x], [y])
    return helper.make_model(graph)


def test_removes_stack_trace_and_keeps_other_metadata(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(_model([_node_with_trace("Identity", ["x"], ["y"])]), path)
    assert SECRET_PATH in path.read_bytes()

    assert strip_export_debug_info(path) == 1

    assert SECRET_PATH not in path.read_bytes()
    node = onnx.load(path).graph.node[0]
    assert [prop.key for prop in node.metadata_props] == ["keep.me"]
    onnx.checker.check_model(onnx.load(path))


def test_removes_stack_trace_from_subgraphs(tmp_path):
    empty = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
    inner = helper.make_graph([_node_with_trace("Identity", ["x"], ["z"])], "inner", [], [empty])
    outer = helper.make_node("If", ["x"], ["y"], then_branch=inner, else_branch=inner)
    path = tmp_path / "model.onnx"
    onnx.save(_model([outer]), path)

    assert strip_export_debug_info(path) == 2
    assert SECRET_PATH not in path.read_bytes()


def test_clean_file_is_not_rewritten(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(_model([helper.make_node("Identity", ["x"], ["y"])]), path)
    before = path.read_bytes()

    assert strip_export_debug_info(path) == 0
    assert path.read_bytes() == before
