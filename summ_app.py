import streamlit as st
from transformers import pipeline
import tensorflow as tf
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import Ollama

# call transformer summarizer model
summarizer = pipeline("summarization", model="./model/")

# add title of app
st.title("Summarization Demo")

# Create text input for user entry
prompt = st.text_area("Enter the text you want to summarize:")

# add model options to app
options = ["Transformer model"] + ["LLM model"]
selection = st.selectbox("Select model option", options=options)

# add parameters to side bar for both transformers and llm models
min_length = st.sidebar.slider("Minimum Summary Length", 10, 200, 50)
max_length = st.sidebar.slider("Maximum Summary Length", 50, 500, 100)
num_predict = st.sidebar.slider("Maximum token generate", 5, 200, 50)
temperature = st.sidebar.slider("Temp", 0, 0, 1)
top_p = st.sidebar.slider("Top_k", 0, 0, 1)

def main():
    
    if st.button("Generate Summary"):
        if prompt:

            if selection == "Transformer model": 
                # Generate summary
                summary = summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=False)
                st.subheader("Generated Summary:")
                st.write(summary[0]["summary_text"])
            else:
                # Generate summary
                llm = OllamaLLM(num_predict=num_predict, temperature=temperature, top_p=top_p, model="phi")
                input_text = f"Summarize the following text:\n\n{prompt}\n\nSummary:"
                response = llm(input_text)
                st.subheader("Generated Summary:")
                st.write(response)

        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
