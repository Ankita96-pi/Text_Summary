import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """
Artificial intelligence (AI) is revolutionizing various industries, from healthcare to finance. It involves the development of algorithms and models that enable machines to perform tasks that typically require human intelligence. Machine learning, a subset of AI, allows systems to learn and improve from experience without explicit programming.
In healthcare, AI is used for diagnosing diseases, predicting patient outcomes, and personalizing treatment plans. In finance, AI algorithms analyze market trends, optimize trading strategies, and detect fraudulent activities. Chatbots powered by AI enhance customer service by providing instant responses and support.
As AI continues to advance, ethical considerations become crucial. Ensuring fairness, transparency, and accountability in AI systems is essential to prevent biases and unintended consequences. Ethical AI practices will shape the responsible and sustainable integration of artificial intelligence into our daily lives.
"""

def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)

    word_frequency = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequency.keys():
                word_frequency[word.text] = 1
            else:
                word_frequency[word.text] += 1

    max_frequency = max(word_frequency.values())
    for word in word_frequency.keys():
        word_frequency[word] = word_frequency[word] / max_frequency

    sentence_tokens = [sentence for sentence in doc.sents]

    sentence_scores = {}
    for sentence in sentence_tokens:
        for word in sentence:
            if word.text in word_frequency.keys():
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = word_frequency[word.text]
                else:
                    sentence_scores[sentence] += word_frequency[word.text]

    select_len = int(len(sentence_tokens) * 0.3)  
    summary = nlargest(select_len, sentence_scores, key=sentence_scores.get)

    final_summary = [word.text for sentence in summary for word in sentence]
    summary = ' '.join(final_summary)
    #print(text)
    #print("Length of original text ",len(text.split(' ')))
    #print("Length of summary text ",len(summary.split(' ')))
    #print(summary)

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))
