"""
Feature extraction for AI vs. human text classification.

All public functions accept a plain string and return a dict of floats.
compute_all_features() is the single entry point for production/UI use.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import nltk
import textstat

# ---------------------------------------------------------------------------
# Lazy singletons — downloaded / loaded once on first use
# ---------------------------------------------------------------------------

def _ensure_nltk():
    for resource, path in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(path, quiet=True)


@lru_cache(maxsize=1)
def _stopwords():
    _ensure_nltk()
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _spacy_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])


@lru_cache(maxsize=1)
def _gpt2_model_and_tokenizer():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    _ensure_nltk()
    return nltk.sent_tokenize(text)


def _word_tokens(text: str) -> list[str]:
    _ensure_nltk()
    return nltk.word_tokenize(text.lower())


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def lexical_features(text: str) -> dict:
    tokens = _word_tokens(text)
    words = [t for t in tokens if t.isalpha()]
    sents = _sentences(text)
    sent_lengths = [len(s.split()) for s in sents]

    n_tokens = max(len(tokens), 1)
    n_words = max(len(words), 1)
    n_sents = max(len(sents), 1)
    sw = _stopwords()

    mean_sl = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    std_sl = float(np.std(sent_lengths)) if sent_lengths else 0.0

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sent_count": n_sents,
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sent_length": mean_sl,
        "std_sent_length": std_sl,
        "max_sent_length": float(max(sent_lengths)) if sent_lengths else 0.0,
        # Burstiness: CV of sentence lengths — high in human text, low in AI
        "burstiness": (std_sl / mean_sl) if mean_sl > 0 else 0.0,
        # Vocabulary richness
        "type_token_ratio": len(set(words)) / n_words,
        "stopword_ratio": sum(1 for w in words if w in sw) / n_words,
        "avg_word_length_chars": float(np.mean([len(w) for w in words])) if words else 0.0,
        # Punctuation ratios (per token)
        "comma_ratio": text.count(",") / n_tokens,
        "period_ratio": text.count(".") / n_tokens,
        "exclamation_ratio": text.count("!") / n_tokens,
        "question_ratio": text.count("?") / n_tokens,
        "semicolon_ratio": text.count(";") / n_tokens,
        "colon_ratio": text.count(":") / n_tokens,
        "quote_ratio": (text.count('"') + text.count("'")) / n_tokens,
        "capitalization_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "paragraph_count": text.count("\n\n") + 1,
    }


def readability_features(text: str) -> dict:
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability": textstat.automated_readability_index(text),
        "dale_chall_score": textstat.dale_chall_readability_score(text),
        "difficult_words_ratio": (
            textstat.difficult_words(text) / max(textstat.lexicon_count(text), 1)
        ),
        "syllable_per_word": (
            textstat.syllable_count(text) / max(textstat.lexicon_count(text), 1)
        ),
    }


def pos_features_from_doc(doc) -> dict:
    total = max(len(doc), 1)
    counts: dict[str, int] = {}
    for token in doc:
        counts[token.pos_] = counts.get(token.pos_, 0) + 1
    tags = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "NUM", "PUNCT"]
    return {f"pos_{p.lower()}_ratio": counts.get(p, 0) / total for p in tags}


def pos_features(text: str) -> dict:
    nlp = _spacy_nlp()
    doc = nlp(text)
    return pos_features_from_doc(doc)


def compute_perplexity(text: str, max_tokens: int = 512) -> float:
    """GPT-2 perplexity. Lower = more AI-like."""
    import torch
    model, tokenizer, device = _gpt2_model_and_tokenizer()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss).cpu())


# ---------------------------------------------------------------------------
# Batch helpers (used by notebook for speed)
# ---------------------------------------------------------------------------

def compute_pos_features_batch(texts: list[str], batch_size: int = 256) -> list[dict]:
    from tqdm.auto import tqdm
    nlp = _spacy_nlp()
    results = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size),
                    total=len(texts), desc="POS tagging"):
        results.append(pos_features_from_doc(doc))
    return results


def compute_perplexity_batch(texts: list[str], batch_size: int = 16) -> list[float]:
    from tqdm.auto import tqdm
    ppls = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GPT-2 perplexity"):
        for text in texts[i: i + batch_size]:
            ppls.append(compute_perplexity(text))
    return ppls


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_features(text: str) -> dict:
    """Compute every feature for a single text string. Used by the UI."""
    feats = {}
    feats.update(lexical_features(text))
    feats.update(readability_features(text))
    feats.update(pos_features(text))
    feats["gpt2_perplexity"] = compute_perplexity(text)
    return feats
