from __future__ import annotations

from transformers import PreTrainedTokenizerBase

REASONER_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer. Respond in the following format:\n"
    "<think>\n...\n</think>\n<answer>\n...\n</answer>"
)

DISCRIMINATOR_SYSTEM_PROMPT = (
    "You are an evaluator responsible for assessing whether a reasoning / thinking process is "
    "reasonable, rigorous, and accurate. Based on these criteria, determine if the analysis is "
    "of high quality. First, analyze the reasoning very briefly, then respond with '**YES**' for "
    "high quality or '**NO**' if it is not. Finally, provide a brief but specific explanation for "
    "your judgment. Hint: You can first summarize the given thinking process to identify the main "
    "reasoning chain, then analyze the reasoning chain sentence by sentence."
)


def _apply_chat(
    tokenizer: PreTrainedTokenizerBase,
    system: str,
    user: str,
) -> str:
    """Format a system+user message pair using the model's own chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for tokenizers without a chat template (rare).
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def reasoner_prompt(question: str, tokenizer: PreTrainedTokenizerBase) -> str:
    return _apply_chat(tokenizer, REASONER_SYSTEM_PROMPT, question)


def discriminator_slice_prompt(
    question: str, slice_text: str, tokenizer: PreTrainedTokenizerBase
) -> str:
    user_msg = (
        f"Question:\n{question}\n\n"
        f"Reasoning slice:\n{slice_text}\n\n"
        "Judge the slice quality."
    )
    return _apply_chat(tokenizer, DISCRIMINATOR_SYSTEM_PROMPT, user_msg)


def discriminator_real_fake_prompt(
    slice_text: str, tokenizer: PreTrainedTokenizerBase
) -> str:
    system = "Classify whether this reasoning slice looks like a high-quality reference slice. Answer YES or NO only."
    user_msg = f"Slice:\n{slice_text}"
    return _apply_chat(tokenizer, system, user_msg)


def discriminator_alignment_prompt(
    slice_text: str, final_correct: bool, tokenizer: PreTrainedTokenizerBase
) -> str:
    marker = "CORRECT" if final_correct else "WRONG"
    system = "Determine whether the slice should be judged as logically sound, given the final answer status. Answer YES or NO only."
    user_msg = f"Final answer status: {marker}\nSlice:\n{slice_text}"
    return _apply_chat(tokenizer, system, user_msg)
