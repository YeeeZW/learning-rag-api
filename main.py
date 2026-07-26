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
    doc_paths = list(Path("docs").glob("*.txt"))

    if not doc_paths
        return None, {}, "docs 文件夹中没有找到 txt 资料"

    documents = []

    for path in doc_paths:
        documents.append(
            {
                "name": path.name,
                "content": path.read_text(encoding="utf-8"),
            }
        )

    texts = [question] + [doc["content"] for doc in documents]

    response = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=texts,
        dimension=1024,
    )

    if response.status_code != HTTPStatus.OK:
        return None, {}, response.message

    question_vector = response.output["embeddings"][0]["embedding"]
    scores = {}

    for index, doc in enumerate(documents, start=1):
        doc_vector = response.output["embeddings"][index]["embedding"]
        scores[doc["name"]] = round(
            cosine_similarity(question_vector, doc_vector),
            4,
        )

    source = max(scores, key=scores.get)

    return source, scores, None


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

    knowledge = (Path("docs") / source).read_text(encoding="utf-8")

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