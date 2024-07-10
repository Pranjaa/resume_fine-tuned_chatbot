import globals
import requests
import streamlit as st
from inference import load_model, generate_response
from process_resume import process_resume

st.set_page_config(page_title="Resume Question-Answering")
st.header("Resume Question Answering")

@st.cache_resource
def initialize_models():
    model_1, tokenizer_1 = load_model(globals.BASE_MODEL_DATASET)
    #model_2, tokenizer_2 = load_model(globals.BASE_MODEL_TRAINING)
    return model_1, tokenizer_1

model_1, tokenizer_1 = initialize_models()
st.write("Initialization done.")

def main():
    question = st.text_input("Ask a question about the documents:")

    with st.sidebar:
        st.subheader("Your documents:")
        uploaded_files = st.file_uploader("Upload the resumes:", type=['txt'], accept_multiple_files=True)
        
        if st.button("Process resumes"):
            if not uploaded_files:
                st.error("Please upload at least one resume file.")
            else:
                with st.spinner("Processing resumes... this might take a while..."):
                    process_resume(uploaded_files, model_1, tokenizer_1)
                st.success("Resumes are processed. Model is retrained.")    

                try:
                    st.success("Resumes uploaded successfully!")

                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")

    if st.button("Submit Question"):
      with st.spinner('Generating Response...'):
        if not question:
            st.error("Please enter a question.")
        else:
            try:
                response = generate_response(question, model_1, tokenizer_1)
                st.write(f"Answer: {response}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

if __name__ == '__main__':
    main()