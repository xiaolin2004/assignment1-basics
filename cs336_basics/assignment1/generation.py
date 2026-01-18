import torch
import torch.nn.functional as F
from cs336_basics.assignment1 import softmax

def generate(
    model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Generate valid next tokens for a given prompt until max_new_tokens is reached
    or eos_token_id is generated.

    Args:
        model: The language model to use for generation.
        prompt_tokens: Tensor of shape (batch_size, sequence_length) containing the prompt tokens.
        max_new_tokens: The maximum number of new tokens to generate.
        eos_token_id: The ID of the end-of-sequence token. If None, generation will not stop early.
        temperature: The temperature value for sampling.
        top_p: The top-p value for nucleus sampling.
    
    Returns:
        Tensor of shape (batch_size, sequence_length + new_tokens) containing the generated tokens.
    """
    model.eval()
    curr_tokens = prompt_tokens.clone()
    
    for _ in range(max_new_tokens):
        # Cropping context if needed happens inside model usually if it handles positional embeddings correctly relative to shape,
        # but for simple implementations we might need to crop input if it exceeds max context.
        # Assuming model handles up to its context length, we might need to trim:
        # idx_cond = curr_tokens if curr_tokens.size(1) <= model.ma else curr_tokens[:, -context_length:]
        # For this assignment, let's assume the user manages context or model handles it (usually we crop to last block_size)
        
        # Taking the last context_length tokens is safe practice if we knew context_length
        # But we don't have it passed here. We'll rely on model or simple forward.
        # Ideally we truncated to model.max_seq_len
         
        with torch.no_grad():
            logits = model(curr_tokens)
        
        # Get logits for the last token position: (B, V)
        next_token_logits = logits[:, -1, :]
        
        # Temperature scaling
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        else:
            # Greedy decoding approximation for temp=0
            # (Strictly speaking we can't divide by 0, so we just take argmax later)
            pass 

        # Top-p (Nucleus) Sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample
        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        # Append
        curr_tokens = torch.cat((curr_tokens, next_token), dim=1)
        
        # Stop condition
        if eos_token_id is not None:
            # If all batch items have generated EOS, we could stop.
            # But simpler logic: if batch_size=1, stop. 
            # For B>1, usually we continue padding or mask. 
            # Let's assume B=1 for simple interactive generation or just break if any/all.
            # Requirement says "until you hit an <|endoftext|>".
            if (next_token == eos_token_id).all():
                break
                
    return curr_tokens
