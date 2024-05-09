from sqlalchemy.orm import Session
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import PromptTemplate
from langchain_postgres.vectorstores import PGVector
from langchain_postgres import PGVector
# from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import Docx2txtLoader
import os
from langchain.tools import tool
from langchain.agents import AgentExecutor,  create_tool_calling_agent,LLMSingleActionAgent,Tool,load_tools,create_structured_chat_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import UnstructuredExcelLoader
from schemas.user import UserQuery 
from pathlib import Path

os.environ['GROQ_API_KEY'] = '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'
os.environ["COHERE_API_KEY"] = ''
os.environ['SERPAPI_API_KEY'] = ''

connection = "postgresql://postgres:postgres@localhost:5432/fastapidb"
embeddings= CohereEmbeddings()
collection_name = "my_docs"
#creating vector store
vectorstore = PGVector(
    embeddings= embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

GROQ_LLM = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
        )

# retriever = vectorstore.as_retriever()
# retrive_tool = create_retriever_tool(
#     retriever,
#     name="Retriever Tool",
#     description="""Use this tool for queries related to cricket""",
# )


# prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#              "system",
#             "You are very powerful assistant",
#         ),
#         ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = load_tools(["serpapi"], llm=GROQ_LLM)

chat_history = []
# tools.append(tool)

def agentmaker():
    llm_with_tools = GROQ_LLM.bind_tools(tools)
    agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
    )
    # agent = (
    # {
    #     "input": lambda x: x["input"],
    #     "agent_scratchpad": lambda x: format_to_openai_tool_messages(
    #         x["intermediate_steps"]
    #     ),
    # }
    # | prompt
    # | llm_with_tools
    # | OpenAIToolsAgentOutputParser()
    # )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

async def helper(file):
    filename=Path(file.filename).stem
    retriever = vectorstore.as_retriever()
    retrive_tool = create_retriever_tool(
    retriever,
    name=f"{filename} Tool",
    description=f"Use this tool for queries related to {filename}",
    )
    tools.append(retrive_tool)
    return "done"
    
async def upload_file_to_db(file:UploadFile):
    temp_path = "documents/"+file.filename
    content = await file.read()
    with open(temp_path,"wb") as temp_file:
        temp_file.write(content)
    print(temp_path)
    file_extension = os.path.splitext(temp_path)[1]
    # if file_extension == '.ppt' or file_extension == '.pptx':
    #    loader = UnstructuredPowerPointLoader(temp_path)
    #    pages= loader.load()
    #    print("padamm",pages)
    #    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    #    splits = text_splitter.split_documents(pages)
    #    print(splits)
    #    vectorstore.add_documents(splits)
    #    return splits
    
    if file_extension == '.docx':
       loader = Docx2txtLoader(temp_path)
       pages= loader.load()
       print("docx",pages)
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
       splits = text_splitter.split_documents(pages)
       print(splits)
       vectorstore.add_documents(splits)
    #    return splits
    elif file_extension == '.pdf':
      loader = PyPDFLoader(temp_path)
      pages = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
      splits = text_splitter.split_documents(pages)
    #   print(splits)
      vectorstore.add_documents(splits)
    elif file_extension == '.ppt' or file_extension == '.pptx':
       loader = UnstructuredPowerPointLoader(temp_path)
       pages= loader.load()
       print("padam",pages)
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
       splits = text_splitter.split_documents(pages)
       print(splits)
       vectorstore.add_documents(splits)
    elif file_extension == '.xls' or file_extension == '.xlsx':
       loader = UnstructuredExcelLoader(temp_path)
    #    loader = UnstructuredExcelLoader("example_data/stanley-cups.xlsx", mode="elements")
       pages = loader.load()
       print("padam",pages)
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
       splits = text_splitter.split_documents(pages)
       print(splits)
       vectorstore.add_documents(splits)
    helperres=await helper(file)
    return "done"

def query_user(query: UserQuery):
    agent_executor=agentmaker()
    # result = agent_executor.invoke({"input": query.question})
    result = agent_executor.invoke({"input": query.question, "chat_history": chat_history})
    chat_history.extend(
    [
        HumanMessage(content=query.question),
        AIMessage(content=result["output"]),
    ]
    )
    print(chat_history)
    return [result['output'],chat_history]