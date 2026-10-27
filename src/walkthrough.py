"""Server-owned, immutable GPT-2 teaching example and pruning presets."""

WALKTHROUGH_VERSION = "population-country-v2"
WALKTHROUGH_SENTENCE = "the most pupulated country in the world is:"
WALKTHROUGH_SEED = 42
WALKTHROUGH_PRESETS = (
    {
        "id": "layers", "title": "Bypass one transformer block", "technique": "layers", "strength": 1,
        "inspection_layer": 11, "inspection_head": 0,
        "explanation": "The last block's output is replaced with its input. Attention and MLP inside that block are still computed, but their contribution is discarded.",
        "inspect": "Select layer 12 and compare its block input and effective output. They become identical on the pruned side.",
    },
    {
        "id": "heads", "title": "Mask three attention heads", "technique": "heads", "strength": 3,
        "inspection_layer": 0, "inspection_head": 0,
        "explanation": "The first three of twelve heads in every layer are zeroed before the output projection. Their attention patterns still exist, but those heads no longer contribute to the residual stream.",
        "inspect": "Inspect head 1's contribution in the computation step: it is zero in the pruned model, even when its attention matrix looks unchanged.",
    },
    {
        "id": "magnitude", "title": "Remove 20% of small MLP weights", "technique": "magnitude", "strength": 20,
        "inspection_layer": 0, "inspection_head": 0,
        "explanation": "Magnitude pruning zeros the lowest-magnitude 20% of each targeted MLP parameter tensor. The MLP transforms token representations differently, which can change later attention and output probabilities.",
        "inspect": "Compare MLP output vectors and the next-token ranking. Input embeddings remain unchanged because they were not pruned.",
    },
    {
        "id": "random", "title": "Remove about 20% of random MLP weights", "technique": "random", "strength": 20,
        "inspection_layer": 0, "inspection_head": 0,
        "explanation": "Random pruning zeros MLP parameters at randomly sampled positions with a fixed seed of 42. Unlike magnitude pruning, the choices do not depend on weight size; the exact fraction can vary slightly.",
        "inspect": "Compare this output with magnitude pruning at the same target sparsity. Same target fraction does not mean the same removed weights or the same prediction.",
    },
)
