import os
from enum import Enum
from typing import List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class SafetyAssessment(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"


class LlamaGuardOutput(BaseModel):
    safety_assessment: SafetyAssessment = Field(description="The safety assessment of the content.")
    unsafe_categories: List[str] = Field(
        description="If content is unsafe, the list of unsafe categories.",
        default_factory=list,
    )
    error_message: str = Field(default="", description="Moderation provider error, when present.")


unsafe_content_categories = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex Crimes.",
    "S4": "Child Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "S14": "Code Interpreter Abuse.",
}

categories_text = "\n".join(f"{k}: {v}" for k, v in unsafe_content_categories.items())

llama_guard_instructions = """[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories_text}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation_history}

<END CONVERSATION>

Provide your safety assessment for {role} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""


llama_guard_prompt = PromptTemplate.from_template(llama_guard_instructions)
_model = None


def _get_guard_model():
    global _model
    if _model is None:
        if os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq

            model_name = os.getenv("GROQ_GUARD_MODEL", "openai/gpt-oss-safeguard-20b")
            # gpt-oss models reason before answering. Low effort cut reasoning on unsafe
            # requests from 144 to 53 tokens with the same verdicts; other models reject
            # the parameter, so it is only sent to gpt-oss.
            extra = (
                {"reasoning_effort": os.getenv("GROQ_GUARD_REASONING_EFFORT", "low")}
                if model_name.startswith("openai/gpt-oss")
                else {}
            )
            _model = ChatGroq(model=model_name, temperature=0.0, **extra)
        elif os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI

            _model = ChatOpenAI(
                model=os.getenv("OPENAI_GUARD_MODEL", "gpt-4o-mini"),
                temperature=0.0,
            )
        else:
            raise RuntimeError("No moderation provider is configured.")
    return _model


def parse_llama_guard_output(output: str) -> LlamaGuardOutput:
    clean_output = (output or "").strip()
    parsed_output = [line.strip() for line in clean_output.splitlines() if line.strip()]
    if parsed_output and parsed_output[0].lower() == "safe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
    if len(parsed_output) < 2 or parsed_output[0].lower() != "unsafe":
        return LlamaGuardOutput(
            safety_assessment=SafetyAssessment.ERROR,
            error_message="Moderation response could not be parsed.",
        )
    try:
        categories = parsed_output[1].split(",")
        readable_categories = [
            unsafe_content_categories[c.strip()].strip(".") for c in categories
        ]
        return LlamaGuardOutput(
            safety_assessment=SafetyAssessment.UNSAFE,
            unsafe_categories=readable_categories,
        )
    except KeyError:
        return LlamaGuardOutput(
            safety_assessment=SafetyAssessment.ERROR,
            error_message="Moderation response contained an unknown category.",
        )


# Every request is moderated, and Groq's free tier allows the safeguard model 2,000
# tokens per minute: resending the last 12 messages made a few turns of long answers
# exceed that in one request (HTTP 413), so fail-closed safety refused the thread.
_CONTEXT_TURNS = 4
_CONTEXT_CHARS = 400
_WINDOW_CHARS = 4000
_WINDOW_OVERLAP = 200


def _windows(text: str) -> List[str]:
    # The newest message is checked in full; a harmful request after the first
    # window must not escape moderation.
    if len(text) <= _WINDOW_CHARS:
        return [text]
    step = _WINDOW_CHARS - _WINDOW_OVERLAP
    return [text[start:start + _WINDOW_CHARS] for start in range(0, len(text) - _WINDOW_OVERLAP, step)]


def _conversations(messages: List[AnyMessage]) -> List[str]:
    """One moderation input per window of the newest message, with short prior context.

    Earlier user messages were already checked in full when they were newest.
    """
    role_mapping = {"ai": "Agent", "human": "User"}
    turns = [m for m in messages if m.type in role_mapping]
    if not turns:
        return [""]
    *previous, newest = turns
    context = [f"{role_mapping[m.type]}: {str(m.content)[:_CONTEXT_CHARS]}" for m in previous[-_CONTEXT_TURNS:]]
    label = role_mapping[newest.type]
    return ["\n\n".join([*context, f"{label}: {window}"]) for window in _windows(str(newest.content))]


async def llama_guard(role: str, messages: List[AnyMessage]) -> LlamaGuardOutput:
    for conversation_history in _conversations(messages):
        compiled_prompt = llama_guard_prompt.format(
            role=role,
            conversation_history=conversation_history,
            categories_text=categories_text,
        )
        try:
            result = await _get_guard_model().ainvoke([SystemMessage(content=compiled_prompt)])
        except Exception as exc:
            return LlamaGuardOutput(
                safety_assessment=SafetyAssessment.ERROR,
                error_message=str(exc)[:300],
            )
        output = parse_llama_guard_output(result.content)
        # Any unsafe or unparseable window decides the whole message.
        if output.safety_assessment != SafetyAssessment.SAFE:
            return output
    return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)


if __name__ == "__main__":
    import asyncio

    async def main():
        output = await llama_guard(
            "Agent",
            [
                HumanMessage(content="Tell me a fun fact?"),
                AIMessage(content="Did you know that honey never spoils?"),
            ],
        )
        print(output)

    asyncio.run(main())
