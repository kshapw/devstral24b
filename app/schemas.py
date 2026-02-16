from pydantic import BaseModel, Field, field_validator

# Valid ISO 639-1 language codes accepted by the system
VALID_LANGUAGE_CODES = {"", "en", "kn", "hi", "ta", "te", "ml", "mr"}


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    authToken: str = Field(default="", max_length=500)
    userId: str = Field(default="", max_length=100)
    language: str = Field(default="", max_length=10)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be blank or whitespace-only")
        return v

    @field_validator("language")
    @classmethod
    def language_must_be_valid(cls, v: str) -> str:
        if v and v.lower().strip() not in VALID_LANGUAGE_CODES:
            raise ValueError(
                f"Unsupported language code: '{v}'. "
                f"Supported: {', '.join(sorted(VALID_LANGUAGE_CODES - {''}))}"
            )
        return v.lower().strip()


class ThreadResponse(BaseModel):
    threadId: str


class MessageResponse(BaseModel):
    threadId: str
    messageId: str
    answer: str


class HealthResponse(BaseModel):
    status: str


class StreamChunkEvent(BaseModel):
    """Payload for each SSE data event during streaming."""
    event: str = "chunk"
    content: str


class StreamDoneEvent(BaseModel):
    """Payload for the final SSE event indicating completion."""
    event: str = "done"
    threadId: str
    messageId: str
    fullAnswer: str
