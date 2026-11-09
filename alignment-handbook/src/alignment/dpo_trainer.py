"""TSR input-format compatibility for the pinned TRL 0.12.2 trainer."""

from trl import DPOTrainer


class TSRDPOTrainer(DPOTrainer):
    """Keep DPO's prompt prefix consistent with SFT and vLLM inference.

    TRL 0.12.2 tokenizes decoder-only prompts without special tokens. Preserve
    its truncation/EOS behavior and prepend BOS only when it is missing. As in
    the verified reproduction patch, this happens after prompt truncation, so
    a 512-token prompt budget can become 513 tokens including BOS. The total
    concatenated-sequence limit is still enforced by TRL.
    """

    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        result = DPOTrainer.tokenize_row(
            features, processing_class, max_prompt_length, max_completion_length, add_special_tokens
        )
        bos = processing_class.bos_token_id
        if bos is not None and (not result["prompt_input_ids"] or result["prompt_input_ids"][0] != bos):
            result["prompt_input_ids"] = [bos] + result["prompt_input_ids"]
        return result
