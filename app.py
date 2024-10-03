import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st

# Load BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


def summarize_text(text):
    if not text or len(text.strip()) < 20:  # Check if the text is too short
        raise ValueError("Text is too short for summarization.")

    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)

    # Check for empty input_ids
    if inputs['input_ids'].size(1) == 0:
        raise ValueError("Tokenization resulted in empty input_ids.")

    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the article
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])

        return article_text
    except Exception as e:
        raise ValueError("Failed to extract text from the URL. Please check the link.") from e


# Streamlit app
def main():
    st.title("News Article Summarizer")

    url = st.text_input("Enter the URL of the news article:")

    if st.button("Summarize"):
        if url:
            try:
                article_text = extract_text_from_url(url)
                if article_text:
                    summary = summarize_text(article_text)
                    st.subheader("Summary:")
                    st.write(summary)
                else:
                    st.write("Could not extract any text from the article.")
            except Exception as e:
                st.write("An error occurred:", e)
        else:
            st.write("Please enter a valid URL.")


if __name__ == "__main__":
    main()
