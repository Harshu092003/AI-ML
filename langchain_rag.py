import os 
import json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import requests

retriever_global = None

def initialize_rag():
    global retriever_global
    persistent_directory = "./chromadb"
    embeddings = HuggingFaceEmbeddings(model = "intfloat/e5-base-v2")
    
    if os.path.exists(persistent_directory):
        print("loading existing vector database")
        vectordb = Chroma(persistent_directory = persistent_directory ,  embedding_function = embeddings)
    else :
        print("creating new vector database ")
        documents = PyPDFDirectoryLoader("pdf/").load()
        
        raw_text = "\n".join([doc.page_content for doc in documents])
        
        patient_data_map = {}
        for doc in documents:
            source = doc.metadata.get("source" , "unknown")
            page = doc.metadata.get("page" , {})
            for part in doc.page_content.split("Patient Name:") :
                part = part.strip()
                if not part:
                    continue
                key = "Patient Name:" + part.split("\n")[0].strip()
                if key not in patient_data_map :
                    patient_data_map[key] = {"pages" : set() , "source" : source}
                patient_data_map[key]["pages"].add(page)
                
        docs = []
        for p in raw_text.split("Patient Name:"):
            p = p.strip()
            if not p :
                continue
            full = "Patient Name" + p
            key = p.split("\n")[0].strip()
            source = patient_data_map.metadata.get("source" , "unknown")
            page = list(patient_data_map.metadata.get("pages" , [])) 
            docs.append(
                Document(
                    page_content = full ,
                    metadata = {"source" : source , "pages" : pages[0] if pages else 0 }
                )
            )
            
        vectordb = Chroma.from_documents(documents = docs , persistent_directory= persistent_directory , embedding = embeddings)
        vectordb.persist()
        
        retriever_global = vectordb.as_retriever(search_type = "similiarity" , search_kwargs = { "k" :3})
        print("RAG initialized Successfully")
        
        
def get_context_and_source(question : str) :
    docs = retriever_global.invoke(f"query : {question}")
    context = "\n\n".join([doc.page_content for doc in docs])
    
    best_docs = max(docs , key = lambda d:
        sum(1 for w in question.lower().split() if w in context.lower().split()),  default= docs[0]
        )
    source = {
        source : best_docs.metadata.get("source", "unknown") ,
        page : best_docs.metadata.get("pages" , [])   
    }
    
    return context , source

def stream_question(question : str) :
    context , source = get_context_and_source(question)
    if retriever_global is None : 
        return RuntimeError("Rag not initailized")
    
    payload = {
        "model" : "llama3" , 
        "stream" : True ,
        "prompt":(
            "use only the given o=context to answer "
            "if answer not found say i dont know "
            f"Context:\n{context}\n\n"
            f"Question:{question}\n\n Answer:"
        ),
        "options" : {"num_predict" : 528 , temperature : 0}     
    }
    
    with requests.post("http://localhost:11434/api/generate" , json = payload , stream = True) as resp : 
        for line in resp.iter_lines():
            if line :
                data = json.load(line)
                token = data.get("response" ,"")
                if token:
                    yield token;
                if data.get("done" , False):
                    yield f"\n\n Source : {source['file']} | Page : {source[page] + 1}"
        
    
    
        

            