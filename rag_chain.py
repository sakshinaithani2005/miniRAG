"""
RAG chain module for Mini RAG app.
Handles prompt engineering and RAG chain construction.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from typing import List


RAG_PROMPT_TEMPLATE = """Answer the question based only on the following contexts. 
Cite sources using [1], [2], etc. at the end of relevant sentences.
If no relevant info, say "No relevant information found".

Contexts:
{context}

Question: {question}
Answer:"""


def format_docs(documents: List[Document]) -> str:
    """
    Format documents for prompt context with numbering and metadata.
    
    Args:
        documents: List of Document objects
    
    Returns:
        Formatted string with numbered sources
    """
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')} (chunk {doc.metadata.get('chunk_id', '?')})"
        for i, doc in enumerate(documents)
    )


def create_rag_chain(
    retriever: ContextualCompressionRetriever,
    llm: ChatGoogleGenerativeAI
):
    """
    Create RAG chain: retriever → format → prompt → LLM → output.
    
    Args:
        retriever: ContextualCompressionRetriever instance
        llm: ChatGoogleGenerativeAI instance
    
    Returns:
        Runnable RAG chain
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def query_rag(
    chain,
    retriever: ContextualCompressionRetriever,
    question: str
) -> tuple:
    """
    Execute RAG query and return answer + retrieved docs.
    
    Args:
        chain: RAG chain from create_rag_chain()
        retriever: ContextualCompressionRetriever instance
        question: User question string
    
    Returns:
        Tuple of (answer_text, retrieved_documents)
    """
    # Generate answer
    answer = chain.invoke(question)
    
    # Retrieve source documents
    retrieved_docs = retriever.invoke(question)
    
    return answer, retrieved_docs
