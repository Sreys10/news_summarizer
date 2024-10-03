import streamlit as st
import requests
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # You can choose another summarizer like LexRank, Luhn, etc.


# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return article_text
    except Exception as e:
        raise ValueError(f"Error extracting text from URL: {e}")


# Summarization using Sumy (LSA in this case)
def summarize_text_with_sumy(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])


# Streamlit app function
def main():
    st.title("News Article Summarizer (Extractive)")

    # Input for URL
    url = st.text_input("Enter the URL of the news article:")

    # Button to Summarize
    if st.button("Summarize"):
        if url:
            try:
                article_text = extract_text_from_url(url)
                # Display extracted text (Optional)
                st.subheader("Extracted Article Text:")
                st.write(article_text)

                # Summarize text using Sumy
                summary = summarize_text_with_sumy(article_text)
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please enter a valid URL.")


if __name__ == "__main__":
    main()
