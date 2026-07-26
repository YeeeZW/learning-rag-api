import os
import math
from http import HTTPStatus
from pathlib import Path

import dashscope
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    length_a = math.sqrt(sum(a * a for a in vector_a))
    length_b = math.sqrt(sum(b * b for b in vector_b))

    return dot_product / (length_a * length_b)


def choose_source(question):
    fastapi_text = Path("docs/fastapi.txt").read_text(encoding="utf-8")
    knowledge_text = Path("docs/knowledge.txt").read_text(encoding="utf-8")

    texts = [question, fastapi_text, knowledge_text]

    response = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=texts,
        dimension=1024,
    )

    if response.status_code != HTTPStatus.OK:
        return None, None, response.message

    question_vector = response.output["embeddings"][0]["embedding"]
    fastapi_vector = response.output["embeddings"][1]["embedding"]
    knowledge_vector = response.output["embeddings"][2]["embedding"]

    fastapi_score = cosine_similarity(question_vector, fastapi_vector)
    knowledge_score = cosine_similarity(question_vector, knowledge_vector)

    scores = {
        "fastapi.txt": round(fastapi_score, 4),
        "knowledge.txt": round(knowledge_score, 4),
    }

    if fastapi_score > knowledge_score:
        return "fastapi.txt", scores, None

    return "knowledge.txt", scores, None


@app.post("/ask")
def ask_question(request: QuestionRequest):
    source, scores, embedding_error = choose_source(request.question)

    if embedding_error:
        return {
            "success": False,
            "model": "text-embedding-v4",
            "source": None,
            "answer": "语义检索调用失败：" + embedding_error,
        }

    knowledge = Path(source).read_text(encoding="utf-8")

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名 Python 学习助手。"
                "只能根据下面的项目资料回答。"
                "如果资料中没有提到，请直接回答：资料中没有提到。\n\n"
                "项目资料：\n"
                + knowledge
            ),
        },
        {
            "role": "user",
            "content": request.question,
        },
    ]

    response = dashscope.Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-plus",
        messages=messages,
        result_format="message",
    )

    if response.status_code == HTTPStatus.OK:
        answer = response.output.choices[0].message.content

        return {
            "success": True,
            "model": "qwen-plus",
            "source": source,
            "scores": scores,
            "answer": answer,
        }

    return {
        "success": False,
        "model": "qwen-plus",
        "source": source,
        "scores": scores,
        "answer": "模型调用失败：" + response.message,
    }