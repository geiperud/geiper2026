import os
import logging
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

app = FastAPI(title="GEIPER AI Cloud Backend")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://geiperud.github.io"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

DOCS_DIR = "documentos"
CHROMA_DIR = "chroma_db"
vectorstore = None
cloud_llm = None

def get_llm():
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        logger.warning("No se encontró el HF_TOKEN en las variables de entorno.")
        return None

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
        logger.error(f"Error cargando LLM de Hugging Face: {e}")
        return None

def init_rag():
    global vectorstore, cloud_llm
    if not HAS_LANGCHAIN:
        logger.warning("Faltan dependencias. Verifica requirements.txt")
        return

    cloud_llm = get_llm()

    # Cargar desde chroma_db/ pre-generado (indexar.py lo crea localmente)
    if os.path.exists(CHROMA_DIR):
        logger.info("Cargando BD vectorial desde chroma_db/ ...")
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.environ.get("HF_TOKEN", ""), model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        logger.info("BD Vectorial cargada desde disco.")
        return

    # Fallback: indexar en vivo desde documentos/ (solo si no existe chroma_db/)
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        logger.info(f"Directorio '{DOCS_DIR}/' creado vacío. Agrega PDFs y corre indexar.py.")
        return

    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()

    if len(docs) == 0:
        logger.info("No hay PDFs en documentos/. RAG no se activará.")
        return

    logger.info(f"Indexando PDFs en vivo ({len(docs)} páginas). Para evitar esto, corre indexar.py primero.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.environ.get("HF_TOKEN", ""), model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    logger.info("BD Vectorial lista.")

@app.on_event("startup")
def on_startup():
    init_rag()

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: str = Field(pattern=r"^(investigacion|tematico)$")

@app.get("/status")
def status():
    return {"status": "ok", "cloud_ready": cloud_llm is not None}

@app.post("/chat")
def chat(request: ChatRequest):
    if cloud_llm is None:
        raise HTTPException(status_code=500, detail="Backend Python sin token a HuggingFace.")

    try:
        if request.mode == "investigacion":
            prompt = ChatPromptTemplate.from_template(
                "Eres el Asistente de Investigación estricto del grupo GEIPER. "
                "Responde en español de forma profesional a esta consulta: {input}"
            )
            chain = prompt | cloud_llm
            response = chain.invoke({"input": request.query})
            return {"response": response}

        elif request.mode == "tematico":
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
                prompt = ChatPromptTemplate.from_template("Responde amablemente en español: {input}")
                chain = prompt | cloud_llm
                response = chain.invoke({"input": request.query})
                return {"response": response}
        else:
            raise HTTPException(status_code=400, detail="Modo inválido.")

    except Exception as e:
        logger.error(f"Error API: {e}")
        raise HTTPException(status_code=500, detail="Servicio temporalmente no disponible.")
