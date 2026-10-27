"""Shared layer selection for pruning and its measured visualizations."""


def bypassed_layer_indices(variant, num_layers):
    if variant is None or variant.technique != "layers":
        return range(0)
    count = min(variant.strength, max(0, num_layers - 1))
    start = 0 if getattr(variant, "layer_position", "last") == "first" else num_layers - count
    return range(start, start + count)
