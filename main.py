# Import Libraries
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
from dotenv import load_dotenv
import PyPDF2
import io

# load the API key from .env file
load_dotenv()

# create the app
app = FastAPI()

# converts text into numbers(searchable)
vectorizer = TfidfVectorizer()

# store document in memory
stored_chunks = []    # actual text pieces
chunk_vectors = None  # number version of those pieces

# connect to Groq AI using API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# splits text into chunks (chunk size-300)
def split_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ENDPOINT 1 - upload a document text/file
@app.post("/ingest")
async def ingest(
    text: Optional[str] = Form(None),   # user can type text directly
    file: Optional[UploadFile] = File(None)  # or upload a file
):
    global stored_chunks, chunk_vectors, vectorizer

    # check user sent something
    if not text and not file:
        return {"error": "Please provide either a text or a file"}

    # if user typed text directly
    if text:
        extracted_text = text

    # if user uploaded a file
    elif file:
        content = await file.read()

        # handle PDF files
        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            extracted_text = ""
            for page in reader.pages:
                extracted_text += page.extract_text()

        # handle plain text files
        else:
            extracted_text = content.decode("utf-8")

    # split the text into small chunks
    chunks = split_into_chunks(extracted_text)
    stored_chunks = chunks

    # convert chunks to numbers and store 
    chunk_vectors = vectorizer.fit_transform(chunks)

    return {"message": f"Ingested {len(chunks)} chunks successfully!"}


# this defines what the /ask request should look like
class Question(BaseModel):
    question: str  # just a question as text


# ENDPOINT 2 - ask a question about the uploaded document
@app.post("/ask")
async def ask(body: Question):
    global stored_chunks, chunk_vectors

    # if user didnt upload the document yet
    if chunk_vectors is None:
        return {"answer": "Please upload a document first using /ingest"}

    # convert the question into numbers 
    question_vector = vectorizer.transform([body.question])

    # check which chunks are most similar to the question
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]

    # grab the top 3 most relevant chunks
    top_indices = np.argsort(similarities)[-3:][::-1]
    relevant_chunks = [stored_chunks[i] for i in top_indices]

    # join the 3 chunks into one block of context
    context = "\n\n".join(relevant_chunks)

    # send the context + question to Groq AI
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                # tell the LLM to ONLY use our document, not its own knowledge
                "content": (
                    "You are a helpful assistant. "
                    "Answer ONLY based on the document context provided below. "
                    "If the answer is not in the context, say 'I don't know based on the provided document.' "
                    "Do not use any outside knowledge.\n\n"
                    f"CONTEXT:\n{context}"
                ),
            },
            # the actual question from the user
            {"role": "user", "content": body.question},
        ],
    )

    # pull the answer text out of the response
    answer = response.choices[0].message.content

    # return the answer + a snippet of what context was used
    return {"answer": answer, "context_used": context[:300] + "..."}


# ENDPOINT 3 - just a health check to confirm app is running
@app.get("/")
def root():
    return {"status": "Document Q&A API is running!"}