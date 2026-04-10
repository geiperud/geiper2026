import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

app = FastAPI(title="GEIPER AI Cloud Backend")

# Permitir CORS para que GitHub Pages pueda conectarse a este servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
vectorstore = None
cloud_llm = None

def get_llm():
    # El Token seguro debe estar guardado en las configuraciones (Secrets) de la cuenta en Hugging Face
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        print("ADVERTENCIA: No se encontró el HF_TOKEN en las variables de entorno.")
        return None
    
    # Conección a Mistral 100% open source
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            huggingfacehub_api_token=api_token,
            temperature=0.3,
            max_new_tokens=500
        )
        return llm
    except Exception as e:
        print(f"Error cargando LLM de Hugging Face: {e}")
        return None

def init_rag():
    global vectorstore, cloud_llm
    if not HAS_LANGCHAIN:
        print("Faltan dependencias. Verifica requirements.txt")
        return

    cloud_llm = get_llm()
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Directorio '{DATA_DIR}' creado vacio. Puedes subir PDFs aquí para el bot temático.")
        return
        
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    
    if len(docs) == 0:
        print("INFO: No hay documentos PDF. RAG no se activará.")
        return
        
    print(f"INFO: Procesando PDFs para RAG. {len(docs)} páginas...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("INFO: BD Vectorial lista.")

@app.on_event("startup")
def on_startup():
    init_rag()

class ChatRequest(BaseModel):
    query: str
    mode: str

@app.get("/status")
def status():
    return {"status": "ok", "cloud_ready": cloud_llm is not None}

@app.post("/chat")
def chat(request: ChatRequest):
    if cloud_llm is None:
        raise HTTPException(status_code=500, detail="Backend Python sin token a HuggingFace.")

    try:
        if request.mode == "investigacion":
            # Bot Investigación
            prompt = ChatPromptTemplate.from_template(
                "Eres el Asistente de Investigación estricto del grupo GEIPER. "
                "Responde en español de forma profesional a esta consulta: {input}"
            )
            chain = prompt | cloud_llm
            response = chain.invoke({"input": request.query})
            return {"response": response}
            
        elif request.mode == "tematico":
            # Bot Temático RAG
            if vectorstore is not None:
                retriever = vectorstore.as_retriever()
                system_prompt = (
                    "Eres el Asistente Temático. Usa el texto recuperado de nuestros documentos para responder.\n\n"
                    "Contexto: {context}\n\nResponde amablemente en español:"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
                qa_chain = create_stuff_documents_chain(cloud_llm, prompt)
                rag_chain = create_retrieval_chain(retriever, qa_chain)
                
                result = rag_chain.invoke({"input": request.query})
                return {"response": result["answer"]}
            else:
                # Sin documentos
                prompt = ChatPromptTemplate.from_template("Responde amablemente en español: {input}")
                chain = prompt | cloud_llm
                response = chain.invoke({"input": request.query})
                return {"response": response}
        else:
            raise HTTPException(status_code=400, detail="Modo inválido.")
            
    except Exception as e:
        print(f"Error API: {str(e)}")
        raise HTTPException(status_code=500, detail="Error enviando datos a HuggingFace. Intenta otra vez.")
