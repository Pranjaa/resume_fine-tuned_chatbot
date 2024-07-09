import requests
import streamlit as st
from inference import generate_response
from process_resume import process_resume
from initialize import initialize, model_1, tokenizer_1

st.set_page_config(page_title="Resume Question-Answering")
st.header("Resume Question Answering")

def main():
    with st.spinner("Initializing..."):
        initialize()
    st.write("Initialization done.")

    question = st.text_input("Ask a question about the documents:")

    with st.sidebar:
        st.subheader("Your documents:")
        uploaded_files = st.file_uploader("Upload the resumes:", type=['txt'], accept_multiple_files=True)
        
        if st.button("Process resumes"):
            with st.spinner("Processing"):
                if not uploaded_files:
                    st.error("Please upload at least one resume file.")
                else:
                    with st.spinner("Processing resumes... this might take a while..."):
                        process_resume(uploaded_files)
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