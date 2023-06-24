import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    XLNetLMHeadModel,
    XLNetTokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer
)

from transformers import set_seed
import time
import argparse
import logging
from typing import List
from tqdm.auto import tqdm

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "distilgpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet-base-cased": (XLNetLMHeadModel, XLNetTokenizer),
    "facebook/opt-125m": (AutoModelForCausalLM, AutoTokenizer),
    "facebook/opt-350m": (AutoModelForCausalLM, AutoTokenizer),
    "facebook/opt-2.7b": (AutoModelForCausalLM, AutoTokenizer),
    "transfo-xl-wt103": (TransfoXLLMHeadModel, TransfoXLTokenizer)
}

def run_benchmark(model, tokenizer, inputs:List[str]):
    # Set padding token id and padding side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    data = inputs * args.num_samples
            
    batch_size = args.batch_size
    total_time = 0
    # Store generated text
    generated_text  = []

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i: i+batch_size]

        # Encode the input sentence
        tokens = tokenizer.encode(batch, return_tensors='pt', padding=True).to(args.device)
        # Create attention mask
        attention_mask = torch.ones_like(tokens)

        # Run initial inference to load model
        _ = model.generate(
            tokens.to(args.device),
            attention_mask=attention_mask.to(args.device),
            max_new_tokens=args.max_tokens,
            num_return_sequences=1
            )
        
        # Measure inference time for the batch
        start_time = time.time()

        # Generate the completed tokens for each batch
        outputs = model.generate(
            tokens.to(args.device),
            attention_mask=attention_mask.to(args.device),
            max_new_tokens=args.max_tokens,
            temperature=2.0, 
            repetition_penalty=2.5,
            do_sample=False
            )
        
        # Measure inference time for the batch
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        # Decode generated tokens
        generated_text = [tokenizer.decode(x) for x in outputs]

        if args.verbose:
            # Print generated texts
            print("\nGenerated Text:")
            for text in generated_text:
                print(text)

    # calculate avg inference time
    avg_time = total_time / len(data)

    # print the benchmark results
    print("\nBenchmark results:")
    print(f"Avg. inference time per sample: {avg_time:.4f} seconds")
    return avg_time

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name",
        default="distilgpt2",
        type=str,
        required=True,
        help="Model name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--max_tokens",
        default=20,
        type=int,
        help="The maximum numbers of tokens to be generated",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="To print the generated text or not",
    )
    parser.add_argument(
        "--num_samples",
        default=100,
        type=int,
        help="The number of inference samples to use for benchmarking",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA"
    )

    global args
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args.seed)

    # Initialize model and tokenizer
    try:
        args.model_name = args.model_name.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
    except Exception:
        raise KeyError(f"The model {args.model_name} you specified is not supported. Supported model names: {', '.join(MODEL_CLASSES.keys())}")

    # Set the logging level to ERROR for the transformers library
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Load the model and tokenizer 
    model = model_class.from_pretrained(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    
    # Move model to appropriate device
    model.to(args.device)

    # usage
    inputs = ["My pet dog Bruce and I are best friends and we love morning walks in central park."]
    
    _ = run_benchmark(model, tokenizer, inputs=inputs)

if __name__ == "__main__":
    main()