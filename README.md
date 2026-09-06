# LSTM + ChatGLM Hybrid Question Answering System

This project implements a Chinese retrieval-augmented question answering (QA) system. A trainable bidirectional LSTM retriever encodes questions and knowledge-base contexts into the same vector space. The most similar contexts are then provided to ChatGLM-6B, which generates the final answer.

The repository contains the training and inference code. The dataset, vocabulary, retriever checkpoint, and ChatGLM weights are not included in the repository; prepare them separately before running the system.

## How It Works

1. Chinese text is segmented with `jieba` and converted to vocabulary IDs.
2. The LSTM retriever applies a bidirectional LSTM, multi-head self-attention, masked mean pooling, and a linear projection.
3. The retriever is trained with an in-batch contrastive (InfoNCE-style) loss.
4. At inference time, all knowledge-base contexts are pre-encoded. A question is matched using cosine similarity and the top-*k* contexts are selected.
5. A prompt containing the selected contexts and the question is sent to ChatGLM-6B.

This design separates retrieval from generation: the LSTM narrows the search space, while ChatGLM produces a natural-language response grounded in the retrieved text.

## Repository Layout

```text
.
|-- lstm_chatglm_hybrid.py   # Retriever, ChatGLM wrapper, and inference CLI
|-- train_hybrid_system.py   # Retriever training and Recall@3 evaluation
|-- data/
|   |-- train.json           # Training question/context pairs (user-provided)
|   |-- validation.json      # Validation pairs (user-provided)
|   |-- test.json            # Optional retrieval evaluation set
|   `-- knowledge_base.json  # Knowledge contexts (user-provided)
`-- lstm_retriever_model/    # Output directory for checkpoints
```

## Requirements

- Python 3.9 or newer
- PyTorch
- Transformers
- `jieba`
- `tqdm`
- NumPy
- A local ChatGLM-6B model directory

Install the Python dependencies with:

```bash
pip install torch transformers jieba tqdm numpy
```

CUDA is strongly recommended for ChatGLM-6B. The scripts automatically select CUDA when it is available and otherwise fall back to CPU; loading a 6B parameter model on CPU can require substantial memory and will be slow.

## Data Format

Training, validation, and test files are JSON arrays. Each item must contain a `question` and its matching `context`:

```json
[
  {
    "id": "example-1",
    "question": "Example question",
    "context": "Knowledge-base passage that answers the question"
  }
]
```

The knowledge base can either be an object with a `knowledge` array or a plain array. In both cases, every entry must contain `context`:

```json
{
  "knowledge": [
    {"context": "First knowledge-base passage"},
    {"context": "Second knowledge-base passage"}
  ]
}
```

The default paths are `data/train.json`, `data/validation.json`, `data/test.json`, and `data/knowledge_base.json`. Use the command-line options to provide different paths.

## Training the Retriever

From the repository root, run:

```bash
python train_hybrid_system.py
```

By default this command:

- builds `vocab.json` from the training, validation, and knowledge-base files;
- trains for 10 epochs with a batch size of 64;
- saves the best checkpoint to `lstm_retriever_model/best_model.pt`;
- saves per-epoch checkpoints as `checkpoint_epoch_<N>.pt`;
- evaluates retrieval with Recall@3 when `data/test.json` is present;
- writes evaluation results to `lstm_retriever_model/evaluation_results.json`.

Example with custom paths and hyperparameters:

```bash
python train_hybrid_system.py \
  --train_data data/train.json \
  --val_data data/validation.json \
  --knowledge_base data/knowledge_base.json \
  --vocab_path vocab.json \
  --output_dir lstm_retriever_model \
  --epochs 10 \
  --batch_size 64 \
  --lr 0.001 \
  --max_len 512
```

The vocabulary and model dimensions must remain compatible when loading a checkpoint. If the vocabulary changes, rebuild the retriever checkpoint as well.

## Running Inference

### Single question

```bash
python lstm_chatglm_hybrid.py \
  --lstm_model lstm_retriever_model/best_model.pt \
  --chatglm_model /path/to/chatglm-6b \
  --vocab vocab.json \
  --knowledge_base data/knowledge_base.json \
  --question "Enter your question here" \
  --top_k 3
```

Use `--no_context` to suppress printing retrieved passages and similarity scores.

### Batch questions

The batch input is a JSON array. Each item needs `question`; `id` is optional:

```json
[
  {"id": "q1", "question": "First question"},
  {"id": "q2", "question": "Second question"}
]
```

Run batch inference with:

```bash
python lstm_chatglm_hybrid.py \
  --chatglm_model /path/to/chatglm-6b \
  --batch_file data/questions.json \
  --output answers.json \
  --top_k 3
```

Each successful output item contains `id`, `question`, `answer`, and `num_contexts`. Failed items are retained with `error: true` so that one bad question does not discard the whole batch.

### Interactive mode

Run the script without `--question` or `--batch_file`:

```bash
python lstm_chatglm_hybrid.py
```

The program then reads questions from standard input. Enter `help` for a short prompt or `q`, `quit`, or `exit` to leave the session.

## Command-Line Options

`lstm_chatglm_hybrid.py` supports:

| Option | Default |描述|
| --- | --- | --- |
| `--lstm_model` | `lstm_retriever_model/best_model.pt` | Retriever checkpoint; the script can continue with random weights if it is missing and you confirm interactively |
| `--chatglm_model` | `zai-org/chatglm-6b` | Local ChatGLM model directory; the current CLI checks that this path exists |
| `--vocab` | `vocab.json` | JSON vocabulary generated during training |
| `--knowledge_base` | `data/knowledge_base.json` | Knowledge-base JSON file |
| `--top_k` | `3` | Number of contexts sent to ChatGLM |
| `--question` | unset | Run one question |
| `--batch_file` | unset | Run questions from a JSON array |
| `--output` | `answers.json` | Batch output path |
| `--max_length` | `512` | Requested answer length parameter |
| `--no_context` | off | Do not print retrieved contexts in single/interactive mode |

## Python API

The main class can also be used from Python:

```python
from lstm_chatglm_hybrid import HybridQASystem

qa = HybridQASystem(
    lstm_model_path="lstm_retriever_model/best_model.pt",
    chatglm_model_path="/path/to/chatglm-6b",
    vocab_path="vocab.json",
    knowledge_base_path="data/knowledge_base.json",
)

result = qa.answer_question(
    "Enter your question here",
    top_k=3,
    return_contexts=True,
)
print(result["answer"])
```

`answer_question` returns the generated `answer`, the original `question`, and the number of retrieved contexts. When `return_contexts=True`, it also returns `contexts` and their cosine-similarity `scores`.

## Model and Data Downloads

The original project points to this external Google Drive folder for supplementary files:

<https://drive.google.com/drive/folders/1872C162NpyaV5iXnHYG4okHhe9fhxIRB?usp=sharing>

ChatGLM-6B weights can also be obtained from the model provider and passed through `--chatglm_model`. Make sure the tokenizer and model files are available in that local directory before starting inference. Although Transformers accepts remote model identifiers, the current CLI preflight requires an existing path.

## Limitations and Practical Notes

- The implementation is tailored to Chinese text and uses `jieba` tokenization.
- Retrieval quality depends on the quality and coverage of the paired question/context data.
- The complete knowledge base is encoded in memory at startup; very large collections may require batching or an approximate nearest-neighbor index.
- The generator is prompted to answer only from the retrieved material, but generated text should still be reviewed for factuality.
- No license file is currently included. 
