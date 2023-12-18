## BALANCED MOE

_MoE can get unbalanced, only dropout regularizes and is not that effective_

**Flow:**
- Run clustering on all embeddings
- MOE for each cluster => train each expert on those 8 and then train the router on the last fold (k-fold)

**Other Ideas:**
- sparse scaling attention heads!