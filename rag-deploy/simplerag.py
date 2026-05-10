import logging
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set the desired logging level (e.g., INFO, DEBUG, WARNING, ERROR)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# Load dos modelos (Embedding e LLM)

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", max_tokens = 200)


# Carregar PDF
def loadData():
    logger.info("Starting data loading process.")
    pdf_link = "arquitetura-design-software.pdf"
    loader = PyPDFLoader(pdf_link, extract_images=False)
    pages = loader.load_and_split()
    logger.info(f"Loaded {len(pages)} pages from PDF.")

    # Separar em Chunks (pedaços de documento)
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 4000, # padrão que normalmente é utlizado no mercado
        chunk_overlap = 20, # evita perda de conteúdo no meio do documento e permite manter o contexto nos pedaços de documento
        length_function = len,
        add_start_index = True
    )

    chunks = text_spliter.split_documents(pages)
    logger.info(f"Split document into {len(chunks)} chunks.")


    # Salvar no Vector DB - Chroma
    vectordb = Chroma.from_documents(chunks, embedding=embedding_model)
    logger.info("Documents saved to Chroma vector store.")

    # Loader Retriever - busca os 3 primeiros documentos relevantes
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever initialized.")

    return retriever

def getRelavantDocuments(question):
    logger.info(f"Getting relevant documents for question: '{question}'")
    retriever = loadData()
    context = retriever.invoke(question)
    logger.info("Relevant documents retrieved.")
    return context


def ask(question, llm):
    logger.info(f"Asking LLM with question: '{question}'")
    TEMPLATE = """
    Você é um Arquiteto de Sistemas senior com experiência em arquitetura de sofware, padrões de projeto
    e boas práticas no desenvolvimento de software. 
    Responda a pergunta abaixo informando o contexto informado.

    Contexto: {context}

    Pergunta: {question}
    """

    prompt = PromptTemplate(input_variables = ['context', 'question'], template = TEMPLATE)
    
    sequence = RunnableSequence(prompt | llm)
    context = getRelavantDocuments(question)
    logger.debug(f"Context provided to LLM: {context}") # Using DEBUG for potentially large context

    response = sequence.invoke({'context': context, 'question': question})
    logger.info("LLM responded successfully.")
    
    return response

def lambda_handler(event, context):
    logger.info(f"Lambda function invoked with event: {event}")
    body = json.loads(event.get('body', {}))
    query = body.get('question')
    if not query:
        logger.error("No 'question' key found in the event.")
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "message": "Bad Request: 'question' parameter is missing."
            })
        }
    logger.info(f"Processing query: '{query}'")
    response = ask(query, llm).content
    logger.info(f"Lambda execution finished. Response length: {len(response)} characters.")
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "message": "Tarefa concluída com sucesso",
            "details": response
        })
    }

