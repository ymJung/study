import os
import fitz  # PyMuPDF

from pptx import Presentation # ppt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from pymilvus import connections
import configparser

config = configparser.ConfigParser()
config.read('config.cfg')
openai_api_key = config['openai']['TOKEN']
VECTOR_HOST = "localhost"
VECTOR_PORT = "19530"

"""
PDF -> text docs
    Chunking -> LangChain ( RecursiveCharacterTextSplitter )
        ADD metadata - {text, source, page} 
"""
def process_pdf(file_path):
 
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    with fitz.open(file_path) as pdf:
        for page in pdf:
            # 페이지 텍스트 추출
            text = page.get_text("text").strip()
            # 청킹: 페이지 내 텍스트를 chunk_size 500, overlap 100로 분할
            chunks = splitter.split_text(text)
            for chunk in chunks:
                metadata = {"source": os.path.basename(file_path), "page": page.number + 1}
                docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

"""
PPT -> text docs
    Chunking -> pptx ( Presentation )  text, shape.text
        ADD metadata {text, source, slide} 
"""
def process_ppt(file_path):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    prs = Presentation(file_path)
    
    for i, slide in enumerate(prs.slides):
        slide_text = ""
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text += shape.text + "\n"
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            for shape in notes_slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text += shape.text + "\n"
        if slide_text.strip():
            chunks = splitter.split_text(slide_text)
            for chunk in chunks:
                metadata = {"source": os.path.basename(file_path), "slide": i + 1}
                docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

"""
store vector db (Milvus)
    embeddings -> HuggingFaceEmbeddings("all-mpnet-base-v2")     
"""
def setup_vector_store(documents):
    # Milvus 서버에 연결 (기본: localhost:19530)
    connections.connect(alias="default", host=VECTOR_HOST, port=VECTOR_PORT)
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Milvus.from_documents(documents, embeddings, collection_name="document_collection")
    return vectorstore

"""
create_retrieval_qa
    vectordb + LLM -> RetrievalQA    
"""
def create_retrieval_qa(vectorstore, llm=None): 
    if llm is None:
        llm = OpenAI(model_name="gpt-4", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa


"""
user qa loop
    query -> qa -> result (answer, source_docs)
    print result
"""
def query_loop(qa):
    print("문서 기반 질의응답 시스템을 시작합니다. (종료하려면 'exit' 입력)")
    while True:
        query = input("\n🔹 질문을 입력하세요: ")
        if query.lower().strip() == "exit":
            print("프로그램을 종료합니다.")
            break

        result = qa({"query": query})
        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

       
        if not source_docs:  # 관련 문서가 없는 경우 일반 응답 처리
            answer = "해당 문서에서 관련 내용을 찾을 수 없습니다. 일반적인 설명을 제공합니다."
            source_str = "General Agent"
        else:
            sources = []
            for doc in source_docs:
                meta = doc.metadata
                if "page" in meta:
                    sources.append(f"{meta['source']} {meta['page']}페이지")
                elif "slide" in meta:
                    sources.append(f"{meta['source']} {meta['slide']}슬라이드")
            source_str = ", ".join(sources)
        
        print("답변:", answer)
        print("출처:", source_str)




def main():
    # 1) PDF, PPT 문서 처리
    documents = [process_pdf(".pdf"), process_pdf(".pdf")]  
    # 2) VECTOR 저장
    vectorstore = setup_vector_store(documents)
    # 3) RAG 생성
    qa = create_retrieval_qa(vectorstore)
    # 4) 질의 실행
    query_loop(qa)

if __name__ == '__main__':
    main()
