## Sparse Attention Heads

_Moe but for attention heads to scale them without crazy compute constraints_

Language models are becoming a part of our everyday life. The underlying architecture behind them, the transformer, analyzes words through its multi-head attention mechanism, which has multiple heads, or channels, that look for specific features of words and phrases in an input. As the number of heads in a model increases (as seen in the larger foundation models of today), the model’s computational complexity increases. However, each head also becomes more specific on what features it searches for in a sequence. Because of this, not all the heads are needed for every computation. The goal of this project is to build and evaluate a new transformer architecture that implements sparse routing to attention layers, which selects only a fraction of attention layers to be computed at one time. The sparse routing mechanism includes a linear router layer and an aggregation mechanism across words in the input within the multi-head attention operation to efficiently select heads to compute. I will train two models with this sparse routing architecture (500 million and 1.5 billion parameters) on a rented A100 to compare to existing counterparts (BERT-large and GPT-2). I will use the Wikipedia and BookCorpus datasets and will compare the models to their counterparts on the MMLU benchmark. This project will bring the AI world closer to trillion-parameter language models without inviable computational requirements.

**Architecture Diagram:**
#
![https://github.com/VisH317/llm-experiments/assets/sar_diagram.png](https://github.com/VisH317/llm-experiments/blob/master/assets/sar_diagram.png)
