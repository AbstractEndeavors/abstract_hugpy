## Part of the Abstract Media Intelligence Platform

This module provides NLP and ML enrichment across the media pipeline.

abstract_hugpy focuses on:
- summarization and keyword extraction
- metadata generation (titles, descriptions, SEO)
- multimodal refinement (text, audio, video)

Full system: https://github.com/AbstractEndeavors/abstract_media_platform

---

## **abstract_hugpy — NLP & Media Enrichment Engine**

A modular NLP and ML layer for transforming extracted media content into **structured, enriched, and decision-ready data**.

Designed to operate as part of a larger pipeline, abstract_hugpy provides:

* summarization
* keyword extraction
* metadata generation
* transcription
* content refinement

---

## 🔹 What This System Does

abstract_hugpy converts raw text and media-derived content into:

* summaries
* keywords and density analysis
* titles and descriptions
* structured metadata
* SEO-ready outputs

It sits **after extraction** and **before storage/publishing** in the pipeline.

---

## 🔹 Core Capabilities

### **Summarization**

* Long-form text summarization (chunked + consolidated)
* Multiple output modes (brief, medium, full)
* Designed for large documents beyond model context limits

---

### **Keyword Extraction (Dual Backend)**

* Transformer-based (KeyBERT) + rule-based (spaCy)
* Preset-driven modes:

  * SEO
  * metadata
  * social
  * long-tail
* Density scoring and keyword classification

---

### **Content Refinement**

* Multi-stage generation:

  * prompt generation (BigBird / LED)
  * refinement via generator model
* Produces:

  * titles
  * descriptions
  * abstracts

---

### **Transcription (Whisper Integration)**

* Audio extraction + transcription pipeline
* Singleton-managed models for reuse and performance 

---

### **Media Metadata Generation**

* Title, keywords, and category derivation from transcripts
* Thumbnail extraction via frame sharpness scoring
* URL generation for media assets

---

## 🔹 Architecture

```text
Raw Text / Transcript
        ↓
Summarization
        ↓
Keyword Extraction
        ↓
Content Refinement
        ↓
Metadata Generation
        ↓
Structured Output
```

---

## 🔹 Key Design Decisions

### **Singleton Model Management**

* models loaded once and reused
* avoids repeated initialization overhead

---

### **Preset-Driven Processing**

* consistent outputs via named configurations
* avoids ad-hoc parameter tuning

---

### **Multi-Backend Strategy**

* combines rule-based + transformer approaches
* ensures fallback and robustness

---

### **Structured Outputs**

* all results returned as typed objects / JSON
* no raw string-only outputs

---

## 🔹 Role in the Platform

abstract_hugpy is the **enrichment layer** of the system:

| Layer          | Module             |
| -------------- | ------------------ |
| Extraction     | abstract_ocr       |
| Structuring    | abstract_pdfs      |
| Video          | abstract_videos    |
| **Enrichment** | **abstract_hugpy** |

---

## 🔹 Why This Exists

Most ML pipelines:

* operate in isolation
* lack structure
* produce inconsistent outputs

abstract_hugpy provides:

* consistent enrichment
* reusable pipelines
* integration with upstream extraction systems

---

## 🔹 Design Philosophy

* **Models are tools, pipelines are systems**
* **Structure over raw output**
* **Consistency over novelty**
* **Enrichment is part of the pipeline, not an afterthought**

---

