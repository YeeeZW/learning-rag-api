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

    if not doc_paths:
        return None, {}, "docs 文件夹中没有找到 txt 资料"

    chunks = []

    for path in doc_paths:
        content = path.read_text(encoding="utf-8")
        paragraphs = content.split("\n\n")

        for chunk_index, paragraph in enumerate(paragraphs, start=1):
            paragraph = paragraph.strip()

            if paragraph:
                chunks.append(
                    {
                        "name": path.name,
                        "chunk_index": chunk_index,
                        "content": paragraph,
                    }
                )

    if not chunks:
        return None, {}, "docs 文件夹中没有有效文本资料"

    texts = [question] + [chunk["content"] for chunk in chunks]

    response = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=texts,
        dimension=1024,
    )

    if response.status_code != HTTPStatus.OK:
        return None, {}, response.message

    question_vector = response.output["embeddings"][0]["embedding"]
    scores = {}
    best_chunk = None
    best_score = -1

    for index, chunk in enumerate(chunks, start=1):
        chunk_vector = response.output["embeddings"][index]["embedding"]
        score = cosine_similarity(question_vector, chunk_vector)
        score_key = f'{chunk["name"]}#chunk{chunk["chunk_index"]}'

        scores[score_key] = round(score, 4)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_score < 0.45:
        return None, scores, None

    return best_chunk, scores, None


@app.post("/ask")
def ask_question(request: QuestionRequest):
    selected_chunk, scores, embedding_error = choose_source(request.question)

    if embedding_error:
        return {
            "success": False,
            "model": "text-embedding-v4",
            "source": None,
            "chunk_index": None,
            "scores": {},
            "answer": "语义检索调用失败：" + embedding_error,
        }

    if selected_chunk is None:
        return {
            "success": True,
            "model": None,
            "source": None,
            "chunk_index": None,
            "scores": scores,
            "answer": "未找到相关资料，请换个问法或补充资料。",
        }

    source = selected_chunk["name"]
    chunk_index = selected_chunk["chunk_index"]
    knowledge = selected_chunk["content"]

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
            "chunk_index": chunk_index,
            "scores": scores,
            "answer": answer,
        }

    return {
        "success": False,
        "model": "qwen-plus",
        "source": source,
        "chunk_index": chunk_index,
        "scores": scores,
        "answer": "模型调用失败：" + response.message,
    }