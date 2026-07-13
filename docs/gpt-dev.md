# `gpt-dev.ipynb` — Let's Build GPT

> The pedagogical companion to Andrej Karpathy's "Let's build GPT" lecture. Builds, from scratch in PyTorch, a character-level **decoder-only Transformer language model** trained on the tinyshakespeare dataset (`input.txt`, ~1.1M chars, 65-token vocabulary). Walks from the simplest possible bigram model up to a full multi-layer, multi-head, positionally-embedded GPT with LayerNorm, residuals, and dropout — exactly mirroring the lecture.

## Lecture Reference
- **Video:** "Let's build GPT: from scratch, in code, spelled out" (Andrej Karpathy, *Neural Networks: Zero to Hero* — Lecture 7).
- **Original notebook:** `nanoGPT/gpt-dev.ipynb` (local, 48 KB).

## Status
**Complete.** The final cell is the full, working, end-to-end GPT and has been executed (outputs are present). No formal exercises or TODOs. `dropout = 0.0` is used (effectively disabled) — typical for the lecture version to keep things stable.

## Concepts Demonstrated
- Character-level tokenization (`stoi`/`itos`, 65-token vocab).
- The autoregressive framing: a single chunk of `block_size+1` tokens yields `block_size` nested (context → target) examples.
- `get_batch(split)` — random minibatch sampling with `batch_size` and `block_size`.
- Bigram baseline: a single `nn.Embedding(vocab_size, vocab_size)` table.
- `F.cross_entropy` loss; the chance-level baseline `-ln(1/65) ≈ 4.87`.
- Autoregressive `generate`: softmax → `multinomial` → concatenate.
- **The mathematical trick in self-attention**: lower-triangular weighted aggregation via matrix multiplication.
- Cumulative average via `tril / row_sums` (version 2) and `zeros + masked_fill(-inf) + softmax` (version 3).
- **Q/K/V self-attention** (version 4): `nn.Linear` projections, `wei = q @ k.transpose(-2,-1)`, masking, softmax, `out = wei @ v`.
- Scaled attention: `wei * head_size**-0.5` to control softmax variance.
- `LayerNorm1d` from scratch (gamma/beta, mean/var over the feature dim).
- Pre-norm residual blocks: `x = x + sa(ln1(x))`, `x = x + ffwd(ln2(x))`.
- `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `BigramLanguageModel` (full GPT).
- Token + learned positional embeddings.
- Context cropping in `generate` (to never exceed positional embedding range).

## Structure (in execution order)
The notebook is sparse on markdown — three markdown cells:
1. **(no heading — initial section is implicit)** — data loading, tokenization, batching, baseline bigram model + training.
2. `### The mathematical trick in self-attention` — weighted aggregation, masking, Q/K/V.
3. *(inline "Notes:" bullet list)* — attention as a communication mechanism, encoder vs. decoder masking, self- vs. cross-attention, scaled attention rationale. Also a small encoder/decoder French→English translation diagram in a code cell.
4. `### Full finished code, for reference` — "You may want to refer directly to the git repo instead though." Introduces the consolidated full GPT cell.

Everything else is code cells with inline `#` comment headers (`# version 1`, `# version 2`, `# version 3: using Softmax`, `# version 4: self-attention!`).

## Detailed Walkthrough

### A. Data preparation
- Read `input.txt`, print length (1,115,394 chars) and a 1000-char sample (Shakespeare dialogue).
- Build vocabulary: `chars = sorted(set(text))`, `vocab_size = 65`, prints the character set.
- `stoi`/`itos` dicts plus `encode`/`decode` lambdas; demonstrates round-trip on `"hii there!"`.
- Tensorize the whole dataset: `data = torch.tensor(encode(text))`, shape `torch.Size([1115394])`.
- Train/val split: 90/10 into `train_data` / `val_data`.

### B. The block / context idea
- Introduces `block_size = 8` and shows how a single chunk of `block_size+1` tokens yields 8 nested (context → target) training examples (the autoregressive framing).
- Defines `get_batch(split)` with `batch_size=4`, `block_size=8`, randomly sampling start offsets and stacking `x` / `y` (shifted by one). Prints the resulting `(4,8)` tensors and enumerates every (context, target) pair.

### C. Baseline: Bigram model (first version)
- `class BigramLanguageModel(nn.Module)` — a single `nn.Embedding(vocab_size, vocab_size)` table; `forward` returns logits `(B,T,C)` and (when targets given) reshapes to `(B*T, C)` / `(B*T)` and computes `F.cross_entropy`. `generate` autoregressively samples with softmax + multinomial and concatenates.
- Instantiates the model, prints `logits.shape` and initial loss (~4.88, which is `-ln(1/65)`).
- Generates 100 chars from a zero seed → untrained gibberish.
- Optimizer: `AdamW(lr=1e-3)`.
- Training loop: 10,000 steps with `batch_size=32`, vanilla `zero_grad → backward → step`. Final loss ~2.38.
- Re-generates 100 chars → now word-like fragments.

### D. The mathematical trick in self-attention
Progressively derives matrix-based weighted aggregation:
- **Toy matrix trick**: lower-triangular `a`, normalized rows, `a @ b` = cumulative averages along time.
- **Toy tensor**: `B,T,C = 4,8,2`, random `x`.
- **Version 1**: explicit double `for` loop computing `xbow[b,t] = mean(x[b,:t+1])`. Correct but O(T²) in Python.
- **Version 2**: `wei = tril / row_sums`, then `xbow2 = wei @ x` — same result, vectorized. Verifies `allclose`.
- **Version 3**: `wei = zeros` masked to `-inf` then `softmax` — generalizes to data-dependent affinities.
- **Version 4: self-attention!**: `nn.Linear` projections for `key`, `query`, `value` (head_size=16, C=32). `wei = q @ k.transpose(-2,-1)`, mask with `tril`, softmax, `out = wei @ v`. Demonstrates the actual Q/K/V mechanism.
- Inspects `wei[0]` — a learned, data-dependent lower-triangular attention pattern.
- **Markdown "Notes" cell**: attention as communication, encoder vs. decoder masking, self- vs. cross-attention, scaled attention.
- **Scaling demonstration**: without `* head_size**-0.5`, `wei` variance grows with `head_size`, sharpening softmax overly; checks `k.var()`, `q.var()`, `wei.var()`. Also `softmax` of a small vector vs. the same ×8.
- **From-scratch `LayerNorm1d`**: gamma/beta, mean/var over feature dim. Verifies post-norm rows have mean≈0, std≈1, while column statistics are not normalized.
- A comment-only encoder/decoder translation template cell.

### E. Full finished code (the reference cell)
A single large cell consolidating everything into a runnable, complete GPT:
- **Hyperparameters**: `batch_size=16`, `block_size=32`, `max_iters=5000`, `eval_interval=100`, `lr=1e-3`, `eval_iters=200`, `n_embd=64`, `n_head=4`, `n_layer=4`, `dropout=0.0`, `device = cuda if available`.
- Re-reads `input.txt`, rebuilds vocab/encode/decode, re-tensorizes, re-splits 90/10.
- `get_batch` with `.to(device)`.
- `estimate_loss()` — `@torch.no_grad()`, eval mode, averages over `eval_iters` batches for both splits.
- **`class Head`** — one self-attention head: K, Q, V linear projections (no bias), registered `tril` buffer, scaled attention (`wei = q @ k.T * C**-0.5`), causal masking, softmax, dropout, `out = wei @ v`.
- **`class MultiHeadAttention`** — `num_heads` `Head`s in parallel, concat along channel dim, project back with `Linear(n_embd, n_embd)` + dropout.
- **`class FeedForward`** — `Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd) → Dropout`.
- **`class Block`** — transformer block: `head_size = n_embd // n_head`, `MultiHeadAttention`, `FeedForward`, two `LayerNorm`s, pre-norm residual updates.
- **`class BigramLanguageModel`** (now a full GPT, name retained): token embedding + **learned positional embedding** (`nn.Embedding(block_size, n_embd)`), a `nn.Sequential` of `n_layer` Blocks, final `LayerNorm`, `lm_head` Linear to vocab. `forward` adds token + positional embeddings, runs through blocks, layer-norms, projects to logits, computes cross-entropy if targets given. `generate` now **crops the context to the last `block_size` tokens** before each forward.
- Instantiates the model, moves to device, prints parameter count.
- AdamW optimizer.
- Training loop: 5000 iterations; every `eval_interval` (100) prints train/val loss via `estimate_loss`.
- After training, generates 2000 chars from a zero seed and prints them.

## Key Results

### Bigram model (pedagogical section)
- Initial loss: **4.8786** (= −ln(1/65), chance).
- After 10,000 steps at `batch_size=32`: loss ≈ **2.3824**.
- Sample post-training (100 tokens): `Wowsa / Wou fe, / INurathoune / IESSARin, / MIOLened sus; / An chin is a arokisu` — recognizable word-ish shapes but no real structure.

### Full GPT (final cell)
- Parameter count: **0.209729 M parameters** (~210K).
- Training runs 5000 iters, printing train/val every 100 steps. Trajectory:
  - step 0: train 4.4116 / val 4.4022
  - step 100: train 2.6568 / val 2.6670
  - step 500: train 2.2964 / val 2.3129
  - step 1000: train 2.1030 / val 2.1302
  - step 2000: train 1.8846 / val 1.9949
  - step 3000: train 1.8000 / val 1.9227
  - step 4000: train 1.7125 / val 1.8623
  - **step 4999: train 1.6664 / val 1.8222**
- Final generated sample (2000 chars) is Shakespeare-like in *form* — speaker labels (`WARICK:`, `LUCENTVO:`, `KING RIVENCE:`, `ROMEO:`), line breaks, pseudo-archaic phrasing — but largely non-semantic / garbled words. This is the expected quality for a ~210K-param char-level model trained briefly on tinyshakespeare.

### Self-attention toy section outputs
- Demonstrates the cumulative-average matrix multiplication result.
- `xbow`, `xbow2`, `xbow3` all `allclose == True`.
- Learned `wei[0]` lower-triangular attention matrix (data-dependent, non-uniform).
- LayerNorm demo: post-norm, `x[0,:].mean() ≈ -3.6e-09`, `x[0,:].std() ≈ 1.0000` (row-normalized), while column stats are not normalized.

## Exercises
No formal exercises or TODOs. No `# TODO`, `# exercise`, or "your code here" markers.

## Caveats / Notes
- Minor pedagogical artifacts remain: the bigram model class is still named `BigramLanguageModel` even in the final GPT version, and a debug `print('logits.shape:', logits.shape)` is left in the *first* (pedagogical) bigram `forward`. These are intentional teaching artifacts, not incompleteness.
- `dropout = 0.0` is used (effectively disabled) — typical for the lecture version to keep things stable.
- Loss curves are not plotted (no matplotlib in the final cell); the loss history is only present as printed text in the cell output.
- The `gpt.py` script in the same folder is the **productionized / cleaned-up script version** of this notebook's final cell (see the directory README for details).

## How to Run
```bash
cd nanoGPT
jupyter notebook gpt-dev.ipynb
# or headless:
jupyter nbconvert --to notebook --execute gpt-dev.ipynb --output gpt-dev.ipynb
```
Dependencies: `torch`, `input.txt` (tinyshakespeare, must be in the working directory). Training the final GPT (5000 iters) takes ~5 minutes on a GPU or ~20 minutes on CPU.
