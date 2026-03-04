#!/usr/bin/env python3
"""
Build a Chinese Q&A dataset where every answer is a 打油诗 (doggerel poem),
using the Gemini API.  Output is a JSONL file that can be fed directly into
sft.py for Qwen2.5 SFT fine-tuning.

Usage
-----
    export GEMINI_API_KEY="your-api-key"

    # Generate answers for all built-in seed questions
    python build_dayo_dataset.py

    # Also let Gemini invent extra questions (total ≈ 150)
    python build_dayo_dataset.py --extra 100

    # Custom output path, model, and thread count
    python build_dayo_dataset.py --output my_dataset.jsonl \\
                                 --model gemini-2.0-flash   \\
                                 --workers 8

Training with sft.py
--------------------
    DATASET_PATH=dayo_dataset.jsonl python sft.py
"""

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen2.5 chat-template helpers
# ---------------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"


def qwen_chat(question: str, answer: str) -> str:
    """Format a single turn in the Qwen2.5 chat template."""
    return (
        f"{IM_START}user\n{question}{IM_END}\n"
        f"{IM_START}assistant\n{answer}{IM_END}"
    )


# ---------------------------------------------------------------------------
# Seed questions — diverse everyday Chinese topics
# ---------------------------------------------------------------------------
SEED_QUESTIONS = [
    # Daily life
    "如何应对工作压力？",
    "如何在城市里找到停车位？",
    "如何快速入睡？",
    "如何戒掉刷手机的坏习惯？",
    "早上赖床怎么办？",
    "如何在拥挤的地铁里保持好心情？",
    "周一上班为什么那么难？",
    "如何与难相处的同事打交道？",
    "下班后如何放松解压？",
    "加班文化该怎么看？",
    # Food & cooking
    "如何煮出好吃的泡面？",
    "减肥期间嘴馋怎么办？",
    "火锅和烧烤哪个更好吃？",
    "如何学会做饭？",
    "奶茶到底能不能多喝？",
    "如何挑选一家好的餐厅？",
    "外卖和自己做饭哪个更划算？",
    # Health
    "如何坚持锻炼身体？",
    "久坐对身体有什么坏处？",
    "如何戒掉熬夜的习惯？",
    "感冒了该怎么办？",
    "如何保持好的视力？",
    # Relationships & society
    "如何向朋友借钱又不尴尬？",
    "相亲靠谱吗？",
    "如何维持长距离恋爱？",
    "父母催婚怎么应对？",
    "如何与家人保持和谐关系？",
    "如何交到真心朋友？",
    # Technology & modern life
    "手机电量不足时该怎么办？",
    "如何防止网络诈骗？",
    "人工智能会取代人类的工作吗？",
    "如何防止个人信息泄露？",
    "社交媒体对生活的影响是什么？",
    # Nature & science
    "为什么天空是蓝色的？",
    "为什么下雨前会闻到泥土香？",
    "冬天为什么那么冷？",
    "夏天为什么那么热？",
    "地球变暖我们能做什么？",
    # Finance
    "如何存第一桶金？",
    "买房还是租房？",
    "如何合理规划个人财务？",
    "如何抵制冲动消费？",
    # Humor / philosophy
    "人生的意义是什么？",
    "摆烂是一种生活态度吗？",
    "如何在平淡的生活中找到乐趣？",
    "失业了该怎么办？",
    "如何面对人生中的失败？",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWER_SYSTEM = """\
你是一位擅长用打油诗回答问题的民间诗人，风格幽默风趣、接地气、朗朗上口。

回答要求：
1. 用4至8行打油诗回答，每行大约5至7个汉字（允许适当变化）。
2. 押韵，读起来顺口。
3. 语气幽默，贴近日常生活。
4. 在诙谐中真正给出有用的建议或观点。
5. 只输出诗的内容，不要任何解释或额外文字。\
"""

QUESTION_GEN_SYSTEM = """\
你是一位擅长提出有趣日常生活问题的创意写手。
请生成 {n} 个适合用打油诗回答的中文问题，涵盖不同话题（生活、食物、健康、情感、科技等）。
要求：每行一个问题，问题以问号结尾，不要编号。\
"""

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def make_model(model_name: str) -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def call_with_retry(model: genai.GenerativeModel, prompt: str,
                    system: str, max_retries: int = 5) -> str:
    """Call the model with exponential-backoff retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [{"role": "user", "parts": [prompt]}],
                generation_config=genai.types.GenerationConfig(temperature=0.9),
                system_instruction=system,
            )
            return response.text.strip()
        except Exception as exc:
            wait = 2 ** attempt + random.uniform(0, 1)
            log.warning("Attempt %d failed (%s). Retrying in %.1fs …", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"All {max_retries} attempts failed for prompt: {prompt[:60]}")


def generate_answer(model: genai.GenerativeModel, question: str) -> str:
    return call_with_retry(model, question, ANSWER_SYSTEM)


def generate_extra_questions(model: genai.GenerativeModel, n: int) -> list[str]:
    """Ask Gemini to invent n fresh questions."""
    raw = call_with_retry(
        model,
        f"请生成{n}个问题。",
        QUESTION_GEN_SYSTEM.format(n=n),
    )
    questions = [q.strip() for q in raw.splitlines() if q.strip().endswith("？") or q.strip().endswith("?")]
    return questions[:n]


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_record(model: genai.GenerativeModel, question: str) -> Optional[dict]:
    try:
        answer = generate_answer(model, question)
        return {"text": qwen_chat(question, answer)}
    except Exception as exc:
        log.error("Failed to generate answer for '%s': %s", question, exc)
        return None


def build_dataset(questions: list[str], model_name: str, workers: int) -> list[dict]:
    model = make_model(model_name)
    records = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build_record, model, q): q for q in questions}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                records.append(result)
            log.info("[%d/%d] completed", done, len(questions))

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 打油诗 Q&A dataset with Gemini for Qwen2.5 SFT."
    )
    parser.add_argument(
        "--output", default="dayo_dataset.jsonl",
        help="Output JSONL file path (default: dayo_dataset.jsonl)",
    )
    parser.add_argument(
        "--extra", type=int, default=0,
        help="Ask Gemini to generate this many additional questions on top of the seed list.",
    )
    parser.add_argument(
        "--model", default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel API worker threads (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    questions = list(SEED_QUESTIONS)

    if args.extra > 0:
        log.info("Generating %d extra questions with Gemini …", args.extra)
        model = make_model(args.model)
        extra = generate_extra_questions(model, args.extra)
        log.info("Got %d extra questions.", len(extra))
        questions.extend(extra)

    # Deduplicate while preserving order
    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)
    questions = unique_questions

    log.info("Building dataset for %d questions using %s …", len(questions), args.model)
    records = build_dataset(questions, args.model, args.workers)

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("Saved %d records to '%s'.", len(records), args.output)
    log.info("To train: DATASET_PATH=%s python sft.py", args.output)


if __name__ == "__main__":
    main()
