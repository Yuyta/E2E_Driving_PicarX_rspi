"""Helpers for the ONNX files produced by training."""

from pathlib import Path
from typing import Iterable, Union

import onnx

# Exporter debug info that records absolute source paths, which reveal the local user name.
STACK_TRACE_KEY = "pkg.torch.onnx.stack_trace"


def _strip_nodes(nodes: Iterable[onnx.NodeProto]) -> int:
    removed = 0
    for node in nodes:
        kept = [prop for prop in node.metadata_props if prop.key != STACK_TRACE_KEY]
        removed += len(node.metadata_props) - len(kept)
        del node.metadata_props[:]
        node.metadata_props.extend(kept)

        for attribute in node.attribute:
            if attribute.HasField("g"):
                removed += _strip_nodes(attribute.g.node)
            for graph in attribute.graphs:
                removed += _strip_nodes(graph.node)
    return removed


def strip_export_debug_info(onnx_path: Union[str, Path]) -> int:
    """Remove exporter stack traces from an ONNX file in place.

    External weight files are left untouched.

    Returns:
        The number of metadata entries removed.
    """
    path = str(onnx_path)
    model = onnx.load(path, load_external_data=False)

    removed = _strip_nodes(model.graph.node)
    for function in model.functions:
        removed += _strip_nodes(function.node)

    if removed:
        onnx.save(model, path)
    return removed
