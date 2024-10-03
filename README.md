# News Summarizer

This project is a **News Summarizer** application built using Python. It supports two types of summarization techniques:

1. **Extractive Summarization** (using Sumy with LexRank)
2. **Abstractive Summarization** (using Pegasus)

The user can either provide raw text or a news article link, and the application will return a summarized version of the content.

## Summarization Techniques

### 1. **Extractive Summarization**:
Extractive summarization involves selecting key sentences from the text itself to form the summary. The LexRank algorithm identifies sentences based on their importance within the context of the document. It focuses on extracting relevant sentences from the document without modifying or rephrasing them.

In this project, the **Sumy** library is used with the **LexRank** algorithm to implement extractive summarization.

### 2. **Abstractive Summarization**:
Abstractive summarization generates new sentences that convey the most critical information from the original text. This approach is closer to how humans summarize and rephrase content. It uses language generation models to create summaries that do not copy-paste the original text directly.

This project uses the **Pegasus** model (by Google) for abstractive summarization, which is one of the state-of-the-art models for text generation tasks.


