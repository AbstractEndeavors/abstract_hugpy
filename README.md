# abstract_hugpy

## Part of the Abstract Media Intelligence Platform

This module provides NLP and ML enrichment across the media pipeline.

`abstract_hugpy` focuses on:

* summarization and keyword extraction
* metadata generation (titles, descriptions, SEO)
* multimodal refinement (text, audio, video)

Full system: [https://github.com/AbstractEndeavors/abstract_media_intelligence](https://github.com/AbstractEndeavors/abstract_media_intelligence)

---

`abstract_hugpy` is a Python module for local-first model orchestration, summarization, keyword extraction, transcription support, and media metadata enrichment.

It began as a Hugging Face-oriented utility layer, but has grown into a broader intelligence module that sits between raw media/text inputs and higher-level content workflows. In practice, it acts as a unified interface for loading models lazily, routing tasks through consistent APIs, and producing structured outputs for downstream systems. 

## Why

Most model wrappers stop at “load model, run inference.”

That is not usually enough in a real pipeline.

Media and document systems need more than raw generation:

* consistent model loading
* reusable singleton lifecycle management
* local path resolution and download control
* summarization presets
* keyword extraction presets
* fallback-friendly dependency handling
* structured outputs for SEO and metadata workflows
* text, audio, video, and PDF-aware utilities

`abstract_hugpy` exists to provide that layer.

## What this module does

### Summarization

* unified summarization API across multiple backends
* preset-driven request handling
* chunked long-text summarization
* lazy-loaded singleton model managers
* configurable input policies for short or invalid text

### Keyword extraction

* transformer-backed extraction with KeyBERT
* rule-based extraction with spaCy
* preset-based post-processing for SEO, metadata, and social use cases
* density analysis, hashtag generation, slug candidates, and filtered keyword tiers

### Model orchestration

* central model registry
* environment-aware path resolution
* lazy local-or-hub source selection
* on-demand snapshot downloads
* reusable singleton-backed managers for expensive models

### Media pipeline support

* Whisper-backed transcription loading
* audio extraction from video
* URL/media helpers
* PDF text analysis for summaries and keyword reports
* media URL derivation for public asset mapping

### Code and generation support

* DeepCoder manager for code generation workflows
* generation config loading
* quantization-aware initialization
* single access point for repeated model usage

## Core design goals

* **local-first when possible**
* **lazy load everything expensive**
* **treat models as managed infrastructure, not loose function calls**
* **return structured results instead of ad hoc tuples/dicts**
* **support larger media/content systems, not just isolated scripts**
* **degrade cleanly when optional dependencies are unavailable** 

## Architecture overview

```text
Raw inputs
├── text
├── audio
├── video
├── pdf text
└── URLs
        │
        v
abstract_hugpy
├── model registry / path resolution
├── lazy dependency import layer
├── singleton model managers
├── summarization backends
├── keyword extraction backends
├── media + seo helpers
└── metadata refinement utilities
        │
        v
Structured outputs
├── summaries
├── keywords
├── hashtags
├── slug candidates
├── transcript text
├── metadata dictionaries
└── seo/media reports
```

## Main module areas

```text
abstract_hugpy/
├── imports/
│   ├── lazy dependency accessors
│   ├── token chunking
│   ├── web/media helpers
│   └── metadata helpers
├── models/
│   ├── model registry and path resolution
│   ├── shared imports/config
│   └── managers/
│       ├── summarizers/
│       ├── keybert_model.py
│       ├── whisper_model.py
│       ├── bigbird_module.py
│       └── deepcoder.py
└── utils/
    └── seo/
        └── pdf_utils.py
```

## Features

### 1. Unified summarization registry

`abstract_hugpy` exposes a summarization registry where multiple backends share one contract.

Current design includes:

* T5-style chunked summarization
* Flan-based summarization
* Falconsai pipeline summarization
* preset-aware request resolution
* request objects for consistent parameter handling

This makes summarization feel like one API even when the underlying model changes.

#### Highlights

* `SummaryRequest`
* `SummaryPreset`
* `InputPolicy`
* backend registry and lookup
* chunk-aware consolidation passes
* short / medium / long summarization modes

## 2. Keyword extraction pipeline

Keyword extraction combines:

* a KeyBERT transformer-based backend
* a spaCy rule-based backend
* a refinement layer that classifies and formats results

This is especially useful because raw keyword extraction is often not enough for production use. The refinement layer turns extracted phrases into:

* primary keywords
* secondary keywords
* dropped terms
* density flags
* meta keyword strings
* hashtags
* slug candidates

#### Presets include

* `seo`
* `metadata`
* `social`
* `long_tail`

## 3. Model registry and storage control

The model config layer provides a registry mapping logical names to:

* hub IDs
* storage folders
* task types
* framework categories

It also supports:

* global model home configuration
* environment-variable overrides
* local path resolution
* lazy download behavior through `snapshot_download`

This keeps model locations predictable and centralized.

## 4. Lazy dependency management

One of the stronger parts of this package is the import layer.

Heavy libraries are not imported directly everywhere. Instead, the module exposes dedicated accessors like:

* `get_torch()`
* `get_transformers()`
* `get_whisper()`
* `get_sentence_transformers()`
* `get_moviepy()`
* `get_pdf2image()`

This matters because it keeps:

* startup lighter
* optional features optional
* failures localized and understandable
* large media/model workflows from eagerly loading things they do not need

The `require()` pattern is especially good here because it forces hard dependency failures to occur at manager initialization time instead of surfacing later as opaque runtime errors.

## 5. Whisper and transcription support

The Whisper manager provides:

* singleton-backed model loading
* configurable model size
* model path override support
* direct transcription wrapper
* audio extraction from video through MoviePy

This makes it useful as the transcription stage in a larger media ingestion pipeline.

## 6. DeepCoder support

The DeepCoder manager extends the module beyond summarization and SEO into code-generation/classic LLM workflows.

It supports:

* lazy singleton initialization
* device-aware loading
* optional quantization
* tokenizer and generation config loading
* chat-template or direct-prompt generation
* output persistence helpers
* model info inspection

That gives the project a broader “ML operations utility layer” identity rather than a narrowly scoped summarization package.

## 7. PDF SEO and document analysis

The PDF utilities provide one of the clearest examples of this module’s role in a larger system.

They support:

* loading full-document text
* page-by-page text loading
* running summarization and keyword extraction per scope
* building structured full-report objects

Outputs are intentionally shaped for downstream consumption rather than only human display.

This makes the package fit naturally into archival, indexing, SEO, and document intelligence workflows.

## 8. Media URL and metadata helpers

The media layer includes filesystem-to-public-URL helpers and scaffolding for deriving metadata from media assets.

That includes functionality such as:

* filesystem path → public asset URL mapping
* basic video metadata derivation flow
* transcript → summary → refined title pipeline
* keyword extraction for media metadata generation

This is the point where `abstract_hugpy` clearly becomes part of a broader platform rather than a standalone model wrapper.

## Installation

```bash
pip install abstract-hugpy
```

Or from source:

```bash
git clone <your-repo-url>
cd abstract_hugpy
pip install -e .
```

## Dependencies

From the package metadata shown, current declared dependencies include:

```text
abstract_utilities
requests
openai_whisper
```

In practice, several optional capabilities also rely on libraries such as:

* transformers
* torch
* sentence_transformers
* keybert
* spacy
* moviepy
* tiktoken
* huggingface_hub
* pydub
* speech_recognition
* pdf2image
* pytesseract
* easyocr
* paddleocr

Because the package uses lazy import accessors, not every environment needs every dependency installed at once. 

## Example usage

### Summarize text

```python
from abstract_hugpy.models.managers.summarizers.summarizers import summarize

text = "Your long-form text goes here..."
result = summarize(text, backend="t5", preset="article")
print(result)
```

### Use a structured summary request

```python
from abstract_hugpy.models.managers.summarizers.summarizers import (
    summarize,
    SummaryRequest,
    InputPolicy,
)

req = SummaryRequest(
    text="Example text to summarize",
    max_chunk_tokens=400,
    summary_mode="medium",
    input_policy=InputPolicy.WARN,
)

print(summarize(request=req, backend="t5"))
```

### Extract refined keywords

```python
from abstract_hugpy.models.managers.keybert_model import refine_keywords

text = "Python-based media enrichment pipeline for summaries, keywords, and SEO metadata."
result = refine_keywords(text, preset="seo")

print(result.primary)
print(result.hashtags)
print(result.meta_keywords)
```

### Transcribe audio with Whisper

```python
from abstract_hugpy.models.managers.whisper_model import whisper_transcribe

transcript = whisper_transcribe(
    audio_path="example.wav",
    model_size="small",
    language="english",
)

print(transcript)
```

### Extract audio from video

```python
from abstract_hugpy.models.managers.whisper_model import extract_audio_from_video

audio_path = extract_audio_from_video("video.mp4", "audio.wav")
print(audio_path)
```

### Use DeepCoder

```python
from abstract_hugpy.models.managers.deepcoder import get_deep_coder

coder = get_deep_coder()
result = coder.generate(
    prompt="Write a Python function that groups strings by length.",
    max_new_tokens=300,
)

print(result)
```

### Analyze a PDF text directory

```python
from abstract_hugpy.utils.seo.pdf_utils import analyze_pdf

report = analyze_pdf("/path/to/pdf_dir")
print(report.to_dict())
```

## Public concepts

A nice strength of this project is that it is moving toward explicit schemas instead of loose return conventions.

Important objects include:

* `ModelConfig`
* `SummaryRequest`
* `SummaryPreset`
* `KeywordRequest`
* `KeywordResult`
* `RefinedResult`
* `PDFSeoResult`
* `PDFSeoReport`

That makes the package easier to compose into larger systems and easier to test.

## Why not just call Hugging Face pipelines directly?

Because the actual problem here is larger than inference.

Direct pipeline calls do not solve:

* lifecycle management
* dependency gating
* model path control
* preset systems
* output normalization
* media-aware utilities
* structured reporting
* consistent backends for multiple tasks

`abstract_hugpy` is the orchestration layer around those concerns.

## Good fit for

* local-first AI utilities
* media enrichment pipelines
* SEO/document intelligence systems
* content analysis workflows
* transcription pipelines
* Python services that need managed access to multiple NLP/ML models
* systems where models should be loaded once and reused

## Less about

* training models
* benchmarking frameworks
* browser-side inference
* pure research experimentation
* single-purpose one-off scripts

This package is more about operational reuse than experimentation.

## Design philosophy

### Lazy by default

Large models and large libraries should not initialize unless they are actually needed.

### Singleton where expensive

If a model is costly to load, repeated calls should reuse it.

### Structured outputs over loose values

Named dataclasses are easier to inspect, serialize, and integrate.

### Presets over repeated argument noise

Common tasks should be easy to invoke consistently.

### Pipeline-aware utilities

Text, audio, video, and PDF workflows should feel connected rather than siloed.

## Current strengths

* coherent model-management layer
* strong lazy import pattern
* reusable summarization registry
* practical keyword extraction and refinement
* real downstream usefulness for SEO/media systems
* good fit for larger platform integration

## Reasonable future split points

You said you will split another day, and that makes sense. When that day comes, the natural seams look like:

* `abstract_hugpy_models`
* `abstract_hugpy_summarizers`
* `abstract_hugpy_keywords`
* `abstract_hugpy_media`
* `abstract_hugpy_seo`

But for now, keeping it unified is still coherent because all of it serves the same higher-level role: media intelligence and model orchestration.

## Roadmap ideas

* CLI entrypoints for summary/keyword/pdf workflows
* stronger public API exports at package root
* richer dependency extras in packaging
* improved test coverage around manager lifecycle
* clearer backend capability matrix
* stricter separation between experimental and stable modules
* normalized metadata interfaces for text/audio/video/PDF

## Status

Actively evolving module for model-backed media intelligence workflows.

Originally closer to a Hugging Face helper layer. Now a broader orchestration package for summarization, keyword extraction, transcription support, SEO analysis, and managed inference inside the Abstract Media Intelligence Platform. 

---

I can also do a second pass that is even more GitHub-polished with badges, a quick-start section, and a “module map” table.
