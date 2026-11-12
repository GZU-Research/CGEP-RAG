"""Generate the final short answer from the selected original passages."""

from typing import Dict

from openai import OpenAI

try:
    from .prompts import build_answer_prompt
    from .triple_extraction import response_usage
except ImportError:
    from prompts import build_answer_prompt
    from triple_extraction import response_usage


class AnswerGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0,
        max_tokens: int = 64,
        api_base: str | None = None,
        top_p: float = 1.0,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate_answer(self, evidence_text: str, question: str) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_answer_prompt(question, evidence_text)}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=["\n"],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The answer-generation response was empty.")
        answer = content.strip().splitlines()[0].strip()
        return {
            "answer": answer,
            "full_response": content,
            "model": self.model,
            **response_usage(response),
        }
