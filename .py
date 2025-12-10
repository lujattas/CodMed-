# import pickle
import streamlit as st
from translate import Translator
from langchain_chroma import Chroma
# from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


@st.cache_resource()
def load_resources():
    embedding_function = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    db = Chroma(persist_directory="./data", embedding_function=embedding_function)
    translator = Translator(to_lang="en")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    # with open("model.pkl", "rb") as f:
    #     text_encoder, classifier = pickle.load(f)
    # text_encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    return db, translator, tokenizer, generator#, text_encoder, classifier


# db, translator, tokenizer, generator, text_encoder, classifier = load_resources()
db, translator, tokenizer, generator = load_resources()

# def get_icd10_code(diagnosis):
#     # تحويل التشخيص إلى تمثيل مناسب باستخدام SentenceTransformer
#     embeddings = text_encoder.transform([diagnosis])
#     # استخدام النموذج للتنبؤ برمز ICD-10
#     icd10_code = classifier.predict(embeddings)[0]
#     return icd10_code


def retrieve_documents(query, top_k=5):
    translated_query = translator.translate(query) # Translate query to English
    docs = db.similarity_search(translated_query, k=top_k) # Search for similar documents in Chroma
    return docs

def generate_answer(query):
    retrieved_docs = retrieve_documents(query)
    # Structure the input text so that it contains the ICD with the diagnosis
    input_text = query + " " + " ".join(f"ICD Code:{doc.metadata['ICDCode']} , Description:{doc.page_content}." for doc in retrieved_docs)
    inputs = tokenizer(input_text, return_tensors='pt')
    summary_ids = generator.generate(inputs['input_ids'], max_length=100, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Display options to the user
    print("Generated Answer:")
    print(generated_text)
    print("\nPlease choose the correct ICD Code from the following options:")
    count = 0
    for idx, doc in enumerate(retrieved_docs):
        count += 1
        print(f"{count}-Option {idx + 1}: ICD Code: {doc.metadata['ICDCode']} , Description: {doc.page_content}")

    # Get user input for the correct ICD code
    while True:
        try:
            user_choice = int(input("\nEnter the option number for the correct ICD Code: ")) - 1
            if user_choice < 0 or user_choice >= len(retrieved_docs):
                raise ValueError("Invalid option number")
            selected_icd_code = retrieved_docs[user_choice].metadata['ICDCode']
            break
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number corresponding to the options listed.")

    return selected_icd_code
