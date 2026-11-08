"""E.D.I.T.H. V13 High-Speed Autoregressive Inference Engine.

Features:
1. Pure PyTorch autoregressive generation with KV caching (O(1) per token).
2. Dynamic int8 quantization support for minimal memory footprint.
3. Streaming token generator yielding text as it generates.
4. Temperature, Top-p (nucleus), and repetition penalty sampling.
5. Stop token detection (<eos>, </edith>, </act>).
"""

import os
import sys
from typing import Dict, Generator, List, Optional, Union

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer


class EdithLMInference:
    def __init__(
        self,
        model: EdithLM,
        tokenizer: EdithByteBPETokenizer,
        device: Optional[str] = None,
        quantize_int8: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

        if quantize_int8:
            print("[Inference] Applying int8 dynamic weight quantization...")
            self.model = self.model.cpu().quantize_int8()
            self.device = "cpu"

        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str = "models/checkpoints/edith_bpe_tokenizer.json",
        device: Optional[str] = None,
        quantize_int8: bool = False,
    ) -> "EdithLMInference":
        """Factory method to load EdithLM from checkpoint and tokenizer with strict validation."""
        full_ckpt_path = os.path.join(PROJECT_ROOT, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
        if not os.path.exists(full_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {full_ckpt_path}")

        full_tok_path = os.path.join(PROJECT_ROOT, tokenizer_path) if not os.path.isabs(tokenizer_path) else tokenizer_path
        if not os.path.exists(full_tok_path):
            raise FileNotFoundError(f"Tokenizer not found at {full_tok_path}")

        tok = EdithByteBPETokenizer(vocab_file=full_tok_path)

        ckpt = torch.load(full_ckpt_path, map_location="cpu", weights_only=False)
        
        # 1. Format Detection and State Dict Extraction
        if not isinstance(ckpt, dict):
            raise TypeError(f"Loaded checkpoint is not a dictionary: {type(ckpt)}")

        ckpt_format = "unknown"
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            ckpt_format = "v15_trainer_dict"
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            ckpt_format = "legacy_trainer_dict"
        elif "tok_embeddings.weight" in ckpt:
            state_dict = ckpt
            ckpt_format = "raw_state_dict"
        else:
            available_keys = list(ckpt.keys())[:10]
            raise ValueError(
                f"Unrecognized checkpoint dictionary structure in {full_ckpt_path}. "
                f"Expected 'model_state_dict', 'model_state', or direct tensor keys. Found: {available_keys}"
            )

        # 2. Architecture Configuration Validation
        if "config" in ckpt and hasattr(ckpt["config"], "dim"):
            config = ckpt["config"]
        elif "tok_embeddings.weight" in state_dict:
            vocab_size, dim = state_dict["tok_embeddings.weight"].shape
            intermediate_dim = state_dict["layers.0.mlp.gate_proj.weight"].shape[0]
            n_layers = len([k for k in state_dict.keys() if k.endswith(".attn_norm.weight")])
            q_dim = state_dict["layers.0.attn.q_proj.weight"].shape[0]
            kv_dim = state_dict["layers.0.attn.k_proj.weight"].shape[0]
            head_dim = 64
            n_heads = q_dim // head_dim if q_dim % head_dim == 0 else 4
            n_kv_heads = kv_dim // head_dim if kv_dim % head_dim == 0 else 2
            config = EdithLMConfig(
                vocab_size=vocab_size,
                max_seq_len=512,
                dim=dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                intermediate_dim=intermediate_dim,
                tie_word_embeddings=True,
            )
        else:
            raise ValueError(f"Unable to resolve EdithLMConfig: 'config' key missing and 'tok_embeddings.weight' not in state_dict.")

        # 3. Tokenizer and Vocabulary Compatibility & Auto-Alignment
        if tok.vocab_size != config.vocab_size:
            # Check for 16k vocabulary auto-alignment
            if config.vocab_size == 16384:
                alt_16k = os.path.join(os.path.dirname(full_tok_path), "edith_bpe_tokenizer_16k.json")
                if os.path.exists(alt_16k):
                    tok = EdithByteBPETokenizer(vocab_file=alt_16k)
            elif config.vocab_size == 3051 or config.vocab_size == 4096:
                alt_v14 = os.path.join(os.path.dirname(full_tok_path), "edith_bpe_tokenizer_v14_3k.json")
                alt_base = os.path.join(os.path.dirname(full_tok_path), "edith_bpe_tokenizer.json")
                if os.path.exists(alt_v14) and config.vocab_size == 3051:
                    tok = EdithByteBPETokenizer(vocab_file=alt_v14)
                elif os.path.exists(alt_base):
                    alt_tok = EdithByteBPETokenizer(vocab_file=alt_base)
                    if alt_tok.vocab_size == config.vocab_size:
                        tok = alt_tok

            # Re-verify after auto-alignment attempt
            if tok.vocab_size != config.vocab_size:
                raise ValueError(
                    f"Tokenizer vocabulary size ({tok.vocab_size}) does not match checkpoint model vocabulary size ({config.vocab_size}). "
                    f"Ensure the correct tokenizer is paired with the checkpoint."
                )

        # 4. Model Instantiation & Strict State Dict Verification
        model = EdithLM(config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Allow missing 'output.weight' only when weights are tied
        if config.tie_word_embeddings and "output.weight" in missing_keys:
            missing_keys = [k for k in missing_keys if k != "output.weight"]

        if unexpected_keys:
            raise ValueError(f"Checkpoint contains unexpected parameter keys: {unexpected_keys[:10]}")
        if missing_keys:
            raise ValueError(f"Checkpoint is missing required parameter weights: {missing_keys[:10]}")

        instance = cls(model=model, tokenizer=tok, device=device, quantize_int8=quantize_int8)
        instance.checkpoint_path = full_ckpt_path
        instance.checkpoint_format = ckpt_format
        instance.config = config
        return instance


    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_tokens: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Streams generated tokens one by one as they are decoded."""
        stop_words = stop_tokens or ["<eos>", "</edith>", "<user>", "</act>"]
        stop_ids = {self.tokenizer.special_token_to_id[w] for w in stop_words if w in self.tokenizer.special_token_to_id}

        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        tokens_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids: List[int] = list(input_ids)
        kv_caches = None

        # Prefill prompt tokens
        logits, _, kv_caches = self.model(tokens_tensor, kv_caches=None, start_pos=0)
        next_token_logits = logits[:, -1, :]

        for step in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for prev_tok in set(generated_ids):
                    if prev_tok < next_token_logits.shape[-1]:
                        if next_token_logits[0, prev_tok] > 0:
                            next_token_logits[0, prev_tok] /= repetition_penalty
                        else:
                            next_token_logits[0, prev_tok] *= repetition_penalty

            # Sample next token
            if temperature <= 0.0:
                next_tok_id = int(torch.argmax(next_token_logits, dim=-1).item())
            else:
                scaled_logits = next_token_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_tok_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_tok_id in stop_ids or next_tok_id == self.tokenizer.eos_id:
                break

            generated_ids.append(next_tok_id)
            token_str = self.tokenizer.decode([next_tok_id])
            yield token_str

            # Next step with KV cache
            next_input = torch.tensor([[next_tok_id]], dtype=torch.long, device=self.device)
            start_pos = len(generated_ids) - 1
            logits, _, kv_caches = self.model(next_input, kv_caches=kv_caches, start_pos=start_pos)
            next_token_logits = logits[:, -1, :]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """Full generation wrapper returning the accumulated string."""
        accumulated = []
        for piece in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        ):
            accumulated.append(piece)
        return "".join(accumulated).strip()
