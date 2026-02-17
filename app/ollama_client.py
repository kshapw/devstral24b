import json as json_mod
import logging
from typing import AsyncIterator, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_MAX_RESPONSE_SIZE = settings.MAX_RESPONSE_SIZE


class OllamaClient:
    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT)
        self.base_url = settings.OLLAMA_URL
        self.llm_model = settings.LLM_MODEL
        self.embed_model = settings.EMBED_MODEL

    def _build_options(self) -> dict:
        """Build Ollama options dict from config."""
        return {
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": settings.LLM_TOP_P,
            "top_k": settings.LLM_TOP_K,
            "repeat_penalty": settings.LLM_REPEAT_PENALTY,
        }

    def _build_messages(
        self, system_prompt: str, history: list[dict] | None, user_message: str
    ) -> list[dict]:
        """Build the messages array for /api/chat."""
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        messages.append({"role": "user", "content": user_message})
        return messages

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict] | None = None,
    ) -> str:
        """Non-streaming chat completion via /api/chat."""
        messages = self._build_messages(system_prompt, history, user_message)
        logger.debug(
            "chat() model=%s messages=%d user_msg_len=%d",
            self.llm_model, len(messages), len(user_message),
        )
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": self._build_options(),
                },
            )
            response.raise_for_status()
            if len(response.text) > _MAX_RESPONSE_SIZE:
                logger.error("Ollama response too large: %d bytes", len(response.text))
                raise ValueError(f"Ollama response exceeds size limit ({len(response.text)} bytes)")
            data = response.json()
            return data["message"]["content"]
        except httpx.HTTPError as e:
            logger.error("Ollama chat request failed: %s", e)
            raise
        except (KeyError, TypeError) as e:
            logger.error(
                "Unexpected Ollama chat response structure: %s — body: %s",
                e, response.text[:500],
            )
            raise

    async def chat_stream(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming chat completion via /api/chat. Yields content chunks."""
        messages = self._build_messages(system_prompt, history, user_message)
        logger.debug(
            "chat_stream() model=%s messages=%d user_msg_len=%d",
            self.llm_model, len(messages), len(user_message),
        )
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "stream": True,
                    "options": self._build_options(),
                },
                timeout=settings.OLLAMA_STREAM_TIMEOUT,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json_mod.loads(line)
                    except json_mod.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed JSON line from Ollama stream: %s",
                            line[:200],
                        )
                        continue
                    if data.get("done"):
                        break
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except httpx.HTTPError as e:
            logger.error("Ollama chat_stream request failed: %s", e)
            raise

    async def classify(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Constrained classification call — deterministic, short output.

        Uses temperature=0 and num_predict=10 to force a single-token
        classification response from the LLM.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        logger.debug(
            "classify() model=%s user_msg_len=%d",
            self.llm_model, len(user_message),
        )
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "top_k": 1,
                        "top_p": 0.1,
                        "num_predict": 10,
                        "repeat_penalty": 1.0,
                    },
                },
            )
            response.raise_for_status()
            if len(response.text) > _MAX_RESPONSE_SIZE:
                logger.error("Ollama classify response too large: %d bytes", len(response.text))
                raise ValueError("Ollama response exceeds size limit")
            data = response.json()
            return data["message"]["content"].strip()
        except httpx.HTTPError as e:
            logger.error("Ollama classify request failed: %s", e)
            raise
        except (KeyError, TypeError) as e:
            logger.error(
                "Unexpected Ollama classify response structure: %s — body: %s",
                e, response.text[:500],
            )
            raise

    async def generate(self, prompt: str) -> str:
        """Legacy non-streaming generate (kept for backward compatibility)."""
        logger.debug(
            "generate() model=%s prompt_len=%d", self.llm_model, len(prompt),
        )
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": self._build_options(),
                },
            )
            response.raise_for_status()
            if len(response.text) > _MAX_RESPONSE_SIZE:
                logger.error("Ollama generate response too large: %d bytes", len(response.text))
                raise ValueError("Ollama response exceeds size limit")
            return response.json()["response"]
        except httpx.HTTPError as e:
            logger.error("Ollama generate request failed: %s", e)
            raise
        except (KeyError, TypeError) as e:
            logger.error(
                "Unexpected Ollama generate response structure: %s — body: %s",
                e, response.text[:500],
            )
            raise

    async def embed(self, text: str) -> list[float]:
        logger.debug(
            "embed() model=%s text_len=%d", self.embed_model, len(text),
        )
        # Try new endpoint first (/api/embed, Ollama >= 0.3.4),
        # fall back to legacy (/api/embeddings) for older versions.
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.embed_model,
                    "input": text,
                },
            )
            if response.status_code == 404:
                # Fall back to legacy endpoint
                response = await self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embed_model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                return response.json()["embedding"]
            response.raise_for_status()
            return response.json()["embeddings"][0]
        except httpx.HTTPError as e:
            logger.error("Ollama embed request failed: %s", e)
            raise
        except (KeyError, TypeError) as e:
            logger.error(
                "Unexpected Ollama embed response structure: %s — body: %s",
                e, response.text[:500],
            )
            raise


default_ollama = OllamaClient()
