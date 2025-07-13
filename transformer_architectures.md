# Transformer Architectures Reference

A concise reference of major Transformer variants, their use-cases, toy examples, and mathematical foundations.

---

## 1. Encoder-Only Models

### Overview
Encoder-only models consist of a stack of self-attention and feed-forward layers, trained to produce contextual embeddings. Pre-training typically uses Masked Language Modeling (MLM) and sometimes Next Sentence Prediction (NSP).

### Use Cases
- Text Classification (e.g., sentiment analysis, topic categorization)  
- Named Entity Recognition (NER)  
- Embedding extraction for Retrieval-Augmented Generation (RAG)  

### Toy Example
Task: Classify “The food was amazing!”  
- Input: `[CLS] The food was amazing! [SEP]`  
- BERT processes and produces a vector at `[CLS]`.  
- A small classifier on `[CLS]` outputs: Positive (0.95), Negative (0.05). → **Positive**

### Mathematical Details

**Self-Attention**  
Given token embeddings \(X \in \mathbb{R}^{n \times d}\):  
\[
\begin{aligned}
Q &= XW^Q, \quad K = XW^K, \quad V = XW^V \\
\text{Attention}(Q,K,V) &= \text{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_k}} + M\Bigr)\,V
\end{aligned}
\]  
- \(M\): mask matrix (\(0\) for visible, \(-\infty\) for masked positions)  
- For MLM, random tokens are replaced by `[MASK]` during pre-training.

**Keywords**: MLM, `[CLS]`, NSP, BERT, RoBERTa, DistilBERT

---

## 2. Decoder-Only Models

### Overview
Decoder-only models are autoregressive: each position attends only to previous tokens via causal masking.

### Use Cases
- Text Generation  
- Code Completion  
- Conversational Agents  

### Toy Example
Prompt: “Once upon a time”  
- Model predicts: “ in a land far away...”

### Mathematical Details

**Causal Masking**  
Define mask \(M\) where  
\[
M_{ij} = 
\begin{cases}
0 & i \ge j,\\
-\infty & i < j.
\end{cases}
\]  
Then  
\[
\text{Attn}(Q,K,V) = \text{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_k}} + M\Bigr)\,V
\]  
ensuring token \(i\) cannot see tokens \(>i\).

**Keywords**: Causal Masking, GPT, Autoregressive, Top-k Sampling

---

## 3. Encoder–Decoder Models

### Overview
Combines an encoder stack for the source sequence and a decoder stack that attends both to itself and to encoder outputs. Pre-training often uses denoising objectives (e.g., span corruption).

### Use Cases
- Machine Translation  
- Summarization  
- Any sequence-to-sequence task  

### Toy Example
Translate: “Bonjour” → “Hello”  
1. Encoder encodes “Bonjour.”  
2. Decoder generates “Hello” one token at a time, attending to encoder outputs.

### Mathematical Details

**Cross-Attention**  
Given encoder outputs \(E\) and decoder inputs \(D\):  
\[
\begin{aligned}
Q_d &= D W^Q, \quad K_e = E W^K, \quad V_e = E W^V,\\
\text{CrossAttn}(Q_d,K_e,V_e) &= \text{softmax}\!\Bigl(\tfrac{Q_d K_e^\top}{\sqrt{d_k}}\Bigr)\,V_e.
\end{aligned}
\]

**Keywords**: T5, BART, PEGASUS, Cross-Attention, Seq2Seq

---

## 4. Efficient & Specialized Variants

- **Longformer**: Sliding-window & global attention for long docs.  
- **BigBird**: Sparse random/global attention.  
- **Reformer**: Locality-sensitive hashing (LSH) attention.  
- **Performer**: Kernel-based linear attention.  
- **Vision Transformer (ViT)**: Applies Transformer to image patches.  

**Keywords**: Sparse Attention, LSH, Sliding Window, Kernelized Attention

---

## 5. Fine-Tuning & Deployment Extensions

- **LoRA** (Low-Rank Adapters): Train small adapter matrices.  
- **Prefix-Tuning / Prompt-Tuning**: Learn soft prompts.  
- **Quantization**: 8-bit, 4-bit for reduced memory & latency.  
- **Knowledge Distillation & Pruning**: Smaller, faster models.  

**Keywords**: LoRA, Quantization, Distillation, Pruning, HuggingFace Trainer

---

## 6. Glossary of Key Terms

- **MLM (Masked Language Modeling)**: Predict masked tokens.  
- **Causal Masking**: Prevent future-token access in self-attention.  
- **Self-Attention**: Tokens attend to each other.  
- **Cross-Attention**: Decoder attends to encoder outputs.  
- **Positional Encoding**: Injects order info.  
- **Autoregressive**: Next-token prediction.  
- **NSP (Next Sentence Prediction)**: Predict if two sentences are consecutive.  
- **Pre-Training / Fine-Tuning**: Two-stage learning.

---

*Use this as your go-to Transformer encyclopedia for design, fine-tuning, and deployment.*  
