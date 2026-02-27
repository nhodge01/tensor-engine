#!/usr/bin/env python3
"""
Universal Sentence Transformer ONNX Exporter
Exports any sentence-transformers model to ONNX with pooling included
"""
import torch
import os
import json
from transformers import AutoTokenizer, AutoModel

# ============== CONFIGURATION ==============
MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 6-layer MiniLM model
BATCH_SIZE = 256
SEQ_LEN = 32
# ==========================================

class EmbeddingModelWithPooling(torch.nn.Module):
    """
    Wrapper that includes the transformer model and pooling layer in a single ONNX graph
    """
    def __init__(self, model_name):
        super().__init__()

        # Load the base transformer model
        self.transformer = AutoModel.from_pretrained(model_name)

        # Load pooling configuration from sentence-transformers format
        pooling_config_path = None
        try:
            # Check if it's a sentence-transformers model with pooling config
            from huggingface_hub import hf_hub_download
            pooling_config_path = hf_hub_download(
                repo_id=model_name,
                filename="1_Pooling/config.json"
            )
        except:
            print("No sentence-transformers pooling config found, using mean pooling by default")

        # Parse pooling configuration
        self.pooling_mode = "mean"  # default
        if pooling_config_path and os.path.exists(pooling_config_path):
            with open(pooling_config_path, 'r') as f:
                pooling_config = json.load(f)
                if pooling_config.get("pooling_mode_cls_token"):
                    self.pooling_mode = "cls"
                elif pooling_config.get("pooling_mode_mean_tokens"):
                    self.pooling_mode = "mean"
                elif pooling_config.get("pooling_mode_max_tokens"):
                    self.pooling_mode = "max"
                print(f"Using pooling mode: {self.pooling_mode}")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass with pooling included
        """
        # Get transformer outputs
        if token_type_ids is not None:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Extract the last hidden states
        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

        # Apply pooling
        if self.pooling_mode == "cls":
            # Use CLS token (first token)
            sentence_embeddings = token_embeddings[:, 0, :]
        elif self.pooling_mode == "mean":
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        elif self.pooling_mode == "max":
            # Max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            sentence_embeddings = torch.max(token_embeddings, dim=1)[0]
        else:
            # Fallback to mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask

        return sentence_embeddings

def export_model_to_onnx(model_name, batch_size=256, seq_len=32):
    """
    Export a HuggingFace model to ONNX with pooling included
    """
    print(f"=== Exporting {model_name} to ONNX ===")

    # Extract model name for directory structure (use part after slash if exists)
    if "/" in model_name:
        model_dir_name = model_name.split("/")[-1]
    else:
        model_dir_name = model_name

    # Use model-specific directory
    output_dir = f"onnx/{model_dir_name}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_pooled.onnx")

    # Load tokenizer to check required inputs
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dummy inputs for tracing
    dummy_text = "This is a test sentence for ONNX export."
    encoded = tokenizer(
        dummy_text,
        padding='max_length',
        max_length=seq_len,
        truncation=True,
        return_tensors="pt"
    )

    # Determine which inputs the model needs
    input_names = []
    dummy_inputs = {}

    # Always need input_ids and attention_mask
    # CRITICAL: Use int32 instead of int64 to save PCIe bandwidth!
    input_names.extend(['input_ids', 'attention_mask'])
    dummy_inputs['input_ids'] = torch.ones(batch_size, seq_len, dtype=torch.int32)
    dummy_inputs['attention_mask'] = torch.ones(batch_size, seq_len, dtype=torch.int32)

    # Check if model uses token_type_ids
    if 'token_type_ids' in encoded:
        input_names.append('token_type_ids')
        dummy_inputs['token_type_ids'] = torch.zeros(batch_size, seq_len, dtype=torch.int32)

    print(f"Model inputs: {input_names}")

    # Load the model with pooling
    print("Loading model with pooling layer...")
    model = EmbeddingModelWithPooling(model_name)
    model.eval()

    # Get the embedding dimension
    with torch.no_grad():
        test_output = model(**{k: v[:1] for k, v in dummy_inputs.items()})
        embedding_dim = test_output.shape[-1]
        print(f"Embedding dimension: {embedding_dim}")

    # Define dynamic axes for batch size and sequence length
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'embeddings': {0: 'batch_size'}
    }

    if 'token_type_ids' in input_names:
        dynamic_axes['token_type_ids'] = {0: 'batch_size', 1: 'sequence_length'}

    # Export to ONNX
    print(f"Exporting to {output_path}...")

    # Convert dummy inputs to tuple for export
    dummy_input_tuple = tuple(dummy_inputs[name] for name in input_names)

    torch.onnx.export(
        model,
        dummy_input_tuple,
        output_path,
        input_names=input_names,
        output_names=['embeddings'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        external_data=False  # Embed weights in the ONNX file
    )

    print(f"✓ SUCCESS: Model exported to {output_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    # Use the fast tokenizer's save method if available
    if hasattr(tokenizer, 'save'):
        tokenizer.save(tokenizer_path)
    else:
        # Save using the standard method
        tokenizer.save_pretrained(output_dir)

    print(f"✓ Tokenizer saved to {output_dir}")

    # Update or create config.json with model parameters
    config_path = "src/config.json"

    # Suggest engine path based on model and parameters
    suggested_engine_path = f"engines/{model_dir_name}/fp16/batch{batch_size}_seq{seq_len}_pooled.engine"

    config = {
        "embedding_dim": int(embedding_dim),
        "max_tokens": seq_len,
        "batch_size": batch_size,
        "engine_path": suggested_engine_path,
        "vocab_path": tokenizer_path if os.path.exists(tokenizer_path) else f"{output_dir}/tokenizer_config.json",
        "model_name": model_name,
        "onnx_path": output_path
    }

    # Check if config exists and update it
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
        existing_config.update(config)
        config = existing_config

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Updated config at {config_path}")
    print(f"\nModel info:")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Max sequence length: {seq_len}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Pooling mode: {model.pooling_mode}")

    return output_path

if __name__ == "__main__":
    # Export the model using the configuration at the top of the file
    export_model_to_onnx(MODEL, BATCH_SIZE, SEQ_LEN)