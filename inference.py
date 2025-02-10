import os
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

def load_llama_model(model_path):
    """Load Llama model and tokenizer"""
    try:
        # Check if model_path exists and is a directory
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path {model_path} must be a directory containing model files")

        # Initialize tokenizer from local path
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Initialize model from local path
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
             rope_scaling={
                "type": "dynamic",
                "factor": 2.0
            }
        )
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def generate_text(model, tokenizer, prompt, 
                 max_length=256,
                 temperature=0.7,
                 top_p=0.95,
                 top_k=50,
                 num_return_sequences=1):
    """Generate text based on prompt"""
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Generate with specified parameters
        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode generated tokens
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in generated_outputs
        ]
        
        return generated_texts
    
    except Exception as e:
        print(f"Error during generation: {e}")
        raise

def main():
    try:
        # Path to your model directory containing the model files
        model_path = "/data/fsx/out/llama-3.2-finetuned/final"
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = load_llama_model(model_path)
        
        # Example prompt
        prompt = "Write a short story about a robot learning to paint:"
        
        # Regular generation
        print("\nGenerating text:")
        generated_texts = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=256,
            temperature=0.7
        )
        for i, text in enumerate(generated_texts):
            print(f"Generated text {i+1}:\n{text}\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
