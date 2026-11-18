"""
01. Introduction to Transformers & the pipeline() API
=====================================================

This lesson introduces the Hugging Face `transformers` library and its
high-level `pipeline()` API — the easiest way to run a transformer model.

The big idea: `pipeline()` is a UNIVERSAL ADAPTER. You change one string
(the task name) and the same function does a completely different job —
text generation, sentiment analysis, question answering, summarization,
and more. The API never changes; only the task string does.

Concepts Covered:
1.  **Pipeline**: A high-level helper that bundles two key components:
    *   **The Tokenizer** (The Translator): Converts text into numbers (tokens) the model understands.
    *   **The Model** (The Brain): The neural network that processes those numbers.
2.  **text-generation**: Continue a prompt with GPT-2.
3.  **sentiment-analysis**: Classify text as positive/negative/neutral (2-class vs 3-class models).
4.  **batch mode**: Process many inputs in one call instead of a loop.
5.  **question-answering** and **summarization**: same API, different task string.

Background — What is a Transformer?

    A Transformer is a neural network architecture invented in 2017 ("Attention Is All You Need").
    Before it, AI read text word-by-word in sequence — slow and forgetful.
    The key innovation: self-attention — every word looks at every other word simultaneously
    and decides "how relevant is each other word to understanding me?"

    Example:
        "The bank by the river was steep"
         ^^^^                              ← is this a financial bank or a riverbank?

        Self-attention lets "bank" look at "river" and figure out it's a riverbank.
        No rule was written for this — the model learned it from billions of examples.

    All modern LLMs (GPT, Claude, Gemini, Llama) are transformer-based.
    The transformer is the "engine" — the library below is how you drive it in Python.

Background — What are "Weights"?

    A neural network is a giant pile of numbers — billions of them — called weights.
    Each weight is a dial that controls how strongly one neuron influences another.

    Training = adjusting all those dials, billions of times, on massive text data,
    until the model gets good at predicting "what word comes next?"

    Once training is done, the dials are frozen — that IS the trained model.
    When you download GPT-2 (~500MB), you're downloading those frozen dials.
    Running the model = feeding your prompt through all those dials to get an answer.

        GPT-2:   117M weights  (~500MB)   — small, fast, runs on a laptop CPU
        GPT-3:   175B weights  (~350GB)   — large, needs big servers
        Claude:  unknown (Anthropic hasn't disclosed)

Libraries:
- `transformers`: The most popular library for working with Large Language Models (LLMs).
- `torch` or `tensorflow`: Deep learning frameworks needed under the hood.
  (We assume PyTorch is installed).

Prerequisites: Python basics. No API key needed — everything runs locally.

How to run:
    python 01_intro_transformers.py

Expected output:
    Section 1: three GPT-2 text completions of a prompt
    Section 2: a 2-class vs 3-class sentiment comparison table
    Sections 3-6: single + batch sentiment, a QA answer, and a summary

Note: The first run downloads several models into ~/.cache/huggingface/
      (GPT-2 ~500MB, plus small sentiment / QA / summarization models).
      Later runs load them from cache and start fast.
"""

from transformers import pipeline as _hf_pipeline  # HuggingFace high-level task API


def pipeline(*args, **kwargs):
    """Thin wrapper around transformers.pipeline() that works on every Mac.

    On some Macs (Apple Silicon + older macOS) two things crash Python hard:
      1. The automatic GPU backend (MPS): "MPSNDArray ... unrecognized selector".
      2. Running on weights that are memory-mapped straight from the cached
         file: "Bus error".
    These models are small, so we (1) force the CPU and (2) copy the weights
    into normal memory. Everything else behaves exactly like pipeline().
    """
    pipe = _hf_pipeline(*args, device="cpu", **kwargs)
    for param in pipe.model.parameters():
        param.data = param.data.clone()
    return pipe


# ── 1. Text Generation (GPT-2) ───────────────────────────────────────────────
# The 'text-generation' task tells the library we want to continue a prompt.
# 'gpt2' is small, old, and CPU-friendly — perfect for learning.
# pipeline() downloads the model on first use and caches it in ~/.cache/huggingface/.
print("=" * 55)
print("SECTION 1 — Text Generation (GPT-2)")
print("=" * 55)
print("Loading GPT-2 (downloads ~500MB on first run)...")
generator = pipeline(task="text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
print(f"\nPrompt: '{prompt}'\nGenerating completions...\n")

# - max_new_tokens=50: generate at most 50 NEW tokens (not counting the prompt).
#   (max_length would count prompt + new together — usually not what you want.)
# - num_return_sequences=3: produce 3 different continuations.
# - pad_token_id: suppresses a harmless warning. GPT-2 has no dedicated "pad"
#   token, so we tell it to use the End-Of-Sentence token as padding filler.
response = generator(
    prompt,
    max_new_tokens=50,
    num_return_sequences=3,
    pad_token_id=generator.model.config.eos_token_id,
)

# The response is a list of dicts, each with a 'generated_text' field.
for idx, res in enumerate(response, start=1):
    print(f"--- Option {idx} ---")
    print(res["generated_text"].strip())
    print()


# ── 2. Sentiment: 2-class vs 3-class ─────────────────────────────────────────
#
# Same pipeline() function, new task string: "sentiment-analysis".
# Sentiment models come in two flavours — choosing the right one matters:
#
#   2-class (binary): POSITIVE or NEGATIVE only.
#     Problem: neutral text gets force-classified into one bucket, often with
#              misleadingly high confidence ("It's fine" → NEGATIVE 0.83).
#              The model literally cannot say "I'm not sure."
#
#   3-class: positive / neutral / negative.
#     Benefit: neutral text lands in the neutral bucket with honest scores.
#
# Rule of thumb: use 3-class for real-world text (reviews, support tickets,
# social posts). Use 2-class only when every input is clearly opinionated.
print("=" * 55)
print("SECTION 2 — Sentiment: 2-class vs 3-class")
print("=" * 55)
print("Loading sentiment models...")
binary_analyzer    = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
print("Ready.\n")

comparison_sentences = [
    "I absolutely love this!",            # clearly positive
    "This is a disaster.",                # clearly negative
    "It's fine, I guess.",                # neutral / lukewarm
    "It exists. It does things.",         # deadpan / ambiguous
    "Not bad, but not great either.",     # mixed
]

print(f"{'Sentence':<40} {'2-class':^22} {'3-class':^22}")
print("-" * 86)
for text in comparison_sentences:
    r2 = binary_analyzer(text)[0]
    r3 = sentiment_analyzer(text)[0]
    label2 = f"{r2['label']} ({r2['score']:.2f})"
    label3 = f"{r3['label']} ({r3['score']:.2f})"
    print(f"{text:<40} {label2:^22} {label3:^22}")

print()
print("Key takeaway:")
print("  Neutral sentences → 2-class forces POSITIVE/NEGATIVE (score often ~1.00)")
print("  Neutral sentences → 3-class returns 'neutral' with an honest lower score\n")


# ── 3. Sentiment: reading a single result ────────────────────────────────────
# Pass one string → get back a list with one dict: {"label": ..., "score": ...}.
# The score is the model's confidence (0.0 – 1.0).
print("=" * 55)
print("SECTION 3 — Sentiment on single sentences")
print("=" * 55)

examples = [
    "I love how easy this library is to use!",
    "The weather is good, but it's not the best.",
    "I'm really disappointed with the service.",
]
for text in examples:
    # TIP: pipeline() always returns a *list*, even for one input. Index [0].
    result = sentiment_analyzer(text)[0]
    # .2f rounds to 2 decimals so scores are easy to compare at a glance.
    print(f"Input  : {text}")
    print(f"Result : {result['label']}  (confidence: {result['score']:.2f})\n")


# ── 4. Sentiment: batch mode ─────────────────────────────────────────────────
# Instead of calling the model N times in a loop, pass a LIST of strings once.
# Why it's faster: the model processes all inputs together in a single pass,
# and you make one round-trip instead of N. On large datasets this is 5–10x faster.
print("=" * 55)
print("SECTION 4 — Batch mode (one call, four sentences)")
print("=" * 55)

batch_texts = [
    "This product exceeded all my expectations.",
    "Absolutely terrible — I want a refund.",
    "It's fine, I guess. Nothing special.",
    "Best purchase I've made all year!",
]
# TIP: pass a list → get a list of dicts back, one per input, in the same order.
batch_results = sentiment_analyzer(batch_texts)
for text, result in zip(batch_texts, batch_results):
    print(f"  [{result['label']:8s}  {result['score']:.2f}]  {text}")
print()


# ── 5. Question Answering ─────────────────────────────────────────────────────
# Same pipeline() function, task = "question-answering".
# TIP: this is *extractive* QA — it finds and returns a span that already exists
#      in the context. It does NOT generate new text.
print("=" * 55)
print("SECTION 5 — Question Answering")
print("=" * 55)
qa_pipeline = pipeline("question-answering")

qa_context = (
    "Artificial Intelligence (AI) is the simulation of human intelligence "
    "in machines that are programmed to think and learn. "
    "Machine Learning is a subset of AI that allows systems to learn from data. "
    "Deep Learning uses neural networks with many layers to process complex patterns."
)
question = "What is Machine Learning?"

# TIP: QA pipeline takes keyword args question= and context= (not positional).
qa_result = qa_pipeline(question=question, context=qa_context)
print(f"Question : {question}")
print(f"Answer   : {qa_result['answer']}  (confidence: {qa_result['score']:.2f})\n")


# ── 6. Summarization ──────────────────────────────────────────────────────────
# Same pipeline() function, task = "summarization".
# TIP: this is *abstractive* — the model generates new sentences, it doesn't
#      just copy spans from the input (the opposite of extractive QA above).
print("=" * 55)
print("SECTION 6 — Summarization")
print("=" * 55)
summarizer = pipeline("summarization")

long_text = (
    "Large language models (LLMs) have transformed natural language processing. "
    "They are trained on enormous datasets scraped from the internet and books, "
    "allowing them to learn grammar, facts, and reasoning patterns. "
    "Models like GPT-4 and Claude can write code, answer questions, translate languages, "
    "and even generate creative fiction. "
    "However, they also inherit biases from their training data and can produce "
    "confident-sounding but incorrect answers — a phenomenon called hallucination. "
    "Researchers are actively working on alignment techniques to make these models "
    "safer and more reliable for real-world applications."
)
# max_length / min_length are in tokens, not words. max_length < input forces compression.
summary_result = summarizer(long_text, max_length=60, min_length=20, do_sample=False)
print(f"Original ({len(long_text.split())} words):")
print(f"  {long_text[:120]}...")
print(f"\nSummary ({len(summary_result[0]['summary_text'].split())} words):")
print(f"  {summary_result[0]['summary_text']}\n")


# ── Recap ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("RECAP — The pipeline() pattern")
print("=" * 55)
print("  pipeline('text-generation')      → freeform text continuation (GPT-2)")
print("  pipeline('sentiment-analysis')   → positive / neutral / negative + score")
print("  pipeline('question-answering')   → extracted answer span")
print("  pipeline('summarization')        → compressed version of long text")
print("  pipeline('translation', model=…) → translated text")
print("  pipeline('ner')                  → named entity tags (Person, Org, …)")
print()
print("Same function. Different task string. That is the HuggingFace pipeline magic.")
