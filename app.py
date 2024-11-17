import streamlit as st  #imported streamlit framework 
from langchain.text_splitter import RecursiveCharacterTextSplitter   #splitting large text documents into smaller
from langchain.document_loaders import PyPDFLoader,DirectoryLoader # it is used to load pdf files and text based files from the directory
from langchain.chains.summarize import load_summarize_chain  # using for summarizing 
from transformers import T5Tokenizer, T5ForConditionalGeneration  # for model evaluation 
from transformers import pipeline # pipe line 
import torch
import base64 

#model and tokenizer 

checkpoint = "LaMini-Flan-T5-248M"
tokenizer =T5Tokenizer.from_pretrained(checkpoint)
basemodel = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype = torch.float32)


# file loader and preprocessing 

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    texts =text_splitter.split_documents(pages)
    final_texts =""
    for text in texts:
        print(text)
        final_texts= final_texts + text.page_content
    return final_texts 

# lm pipe line 

def llm_pipeline(filemain):
    pipe_sum= pipeline(
        'summarization',
        model=basemodel,
        tokenizer=tokenizer,
        max_length = 500,
        min_length =50


    )

    input_text = file_preprocessing(filemain)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

#stream lit code 

@st.cache_data


def displaypdf(file):
    # Read the file in binary mode
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')  # Encode to Base64

    # Embedding the PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Display the PDF file in Streamlit
    st.markdown(pdf_display, unsafe_allow_html=True)




#streamlit code 

st.set_page_config(layout="wide",page_title="summarization App")

def main():
    st.title("Text Summarization using LLM")

    uploaded_file = st.file_uploader('upload your PDF file',type=['pdf'])

    if uploaded_file is not None:
        if st.button("summarize"):
            col1,col2 = st.columns(2)
            filepath="data/"+uploaded_file.name
            with open(filepath,'wb') as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("uploaded PDF file")
                pdf_viewer = displaypdf(filepath)

            with col2:
                st.info("summarization is below")
                summary=llm_pipeline(filepath)
                st.success(summary)




if __name__ == '__main__':
    main()