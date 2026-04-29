#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

import os
import json

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# Load dos modelos (Embedding e LLM)

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", max_tokens = 200)


# Carregar PDF
def loadData():
    pdf_link = "Clean-Architecture.pdf"
    loader = PyPDFLoader(pdf_link, extract_images=False)
    pages = loader.load_and_split()

    # Separar em Chunks (pedaços de documento)
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 4000, # padrão que normalmente é utlizado no mercado
        chunk_overlap = 20, # evita perda de conteúdo no meio do documento e permite manter o contexto nos pedaços de documento
        length_function = len,
        add_start_index = True
    )

    chunks = text_spliter.split_documents(pages)


    # Salvar no Vector DB - Chroma
    vectordb = Chroma.from_documents(chunks, embedding=embedding_model)

    # Loader Retriever - busca os 3 primeiros documentos relevantes
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    return retriever

def getRelavantDocuments(question):
    retriever = loadData()
    context = retriever.invoke(question)
    return context


def ask(question, llm):
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

    response = sequence.invoke({'context': context, 'question': question})
    
    return response

def lambda_handler(event, context):
    query = event.get('question')
    response = ask(query, llm).content
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

