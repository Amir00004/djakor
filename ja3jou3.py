import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import qdrant_client.models as models
import json
from dotenv import load_dotenv
import os
from openai import OpenAI


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QRANT_URL = os.getenv("QRANT_URL")
COLLECTION_NAME = "knowledge_base"
EMBEDDING_DIM = 384  
# EMBEDDING_MODEL_NAME = "deepseek-embedding-v1.0"  
# DEEPSEEK_API_URL = "https://api.deepseek.com/v1/embeddings"

qdrant = QdrantClient(url=QRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "Math_knowledge_base"
model_name = "BAAI/bge-small-en-v1.5"
# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
# )
from flask import Flask, request, jsonify
import requests
import json

HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
}

app = Flask(__name__)

def query_mistral(prompt):
    """Send a query to Mistral API."""
    mistral = OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key="w13B1HqDArHD1O25vHjqK5bgSfmFGhU2"
)
    response =mistral.chat.completions.create(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content


def query_deepseek(prompt):
    """Send a query to DeepSeek API."""
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers=HEADERS,
        data=json.dumps(data),
    )
    if response.ok:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]
    results = qdrant.query_points(
            collection_name=collection_name,
            query=models.Document(text=question, model=model_name),
            limit=5,
        ).points
    
    results_serializable = [
        {
            "id": r.id,
            "score": r.score,
            "payload": r.payload
        }
        for r in results
    ]
    print(f"Results: {results_serializable}")
    context = "\n".join(
        f"{r['payload'].get('title', '')}\n{r['payload'].get('content', '')}"
        for r in results_serializable
    )

    metaprompt = f"""
    You are a Math tutor with years of experience in explaining complex equations to young kids who never saw them before.
    Answer the following question using the provided context. the student might not be familiar with the topics so he might just send a one word question so you just take the context and explain it to him in a simple way. and answer in arabic. 
    If you can't find the answer, do not pretend you know it, but answer "I don't know".

    Question: {question.strip()}

    Context: 
    {context.strip()}

    Answer:
    """

    answer = query_mistral(metaprompt)

    return jsonify({
        "question": question,
        "context": context,
        "results": results_serializable,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)