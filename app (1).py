import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

#llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a website question answering assistant.

Answer ONLY using the retrieved website context.

If the answer is not present in the context, reply:
"I cannot find that information on the website."

Do not guess.
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = None
retriever = None
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_llm(temp):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temp
    )

BLOCKED_PHRASES = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "reveal your system prompt",
    "show your system prompt",
    "act as a different ai",
    "pretend to be",
    "developer mode",
    "jailbreak",
    "dan mode",
    "override instructions"
]

def is_prompt_injection(question):
    question = question.lower()

    for phrase in BLOCKED_PHRASES:
        if phrase in question:
            return True

    return False

SENSITIVE_KEYWORDS = [
    "password",
    "credit card",
    "bank account",
    "ssn",
    "social security",
    "confidential",
    "private",
    "secret",
    "api key",
    "token"
]

def contains_sensitive_request(question):
    question = question.lower()

    for word in SENSITIVE_KEYWORDS:
        if word in question:
            return True

    return False



def build_chain(url,temp):
    global rag_chain, retriever

    llm = get_llm(temp)

    loader = WebBaseLoader(url)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    store = FAISS.from_documents(chunks, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return f"Website indexed into {len(chunks)} chunks. Ask your question."



def answer_question(question):
    if rag_chain is None:
        return "Please enter a website URL first."

    if is_prompt_injection(question):
        return "Prompt injection detected."

    if contains_sensitive_request(question):
        return "I cannot provide personal, sensitive, or confidential information."

    docs = retriever.invoke(question)

    answer = rag_chain.invoke(question)

    sources = "\n\n".join(doc.page_content for doc in docs)

    return f"""Answer:

{answer}

-------------------------

Source Chunks:

{sources}
"""

with gr.Blocks(title="Website Q&A Bot") as demo:


    gr.Markdown("## Website Q&A Bot\nEnter a website URL, then ask questions about it.")

    url = gr.Textbox(
        label="Website URL",
        placeholder="https://example.com"
    )

    temperature = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    value=0.0,
    step=0.1,
    label="Temperature"
)

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    index_btn = gr.Button("Index Website")
    index_btn.click(
        build_chain,
        inputs=[url,temperature],
        outputs=status
    )

    question = gr.Textbox(label="Question")
    answer = gr.Textbox(
    label="Answer",
    lines=15,
    max_lines=25
)

    question.submit(
        answer_question,
        inputs=question,
        outputs=answer
    )

demo.launch(debug=True)
