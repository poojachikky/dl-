from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer():
    """
    Load the GPT-2 model and tokenizer for creative content generation.
    """
    print("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def generate_creative_content(prompt, model, tokenizer, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate creative content based on the user-provided prompt.

    Args:
        prompt (str): User input for text generation.
        model: Pre-trained GPT-2 model.
        tokenizer: GPT-2 tokenizer.
        max_length (int): Maximum length of the generated text.
        temperature (float): Sampling temperature (controls creativity).
        top_k (int): Top-k sampling for diversity.
        top_p (float): Top-p sampling for diversity.

    Returns:
        str: Generated creative content.
    """
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the model
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True
    )

    # Decode the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if _name_ == "_main_":
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    print("\nWelcome to the Creative Content Generator!")
    prompt = input("Enter a prompt to inspire creativity: ")

    # Generate creative content
    print("\nGenerating creative content...\n")
    creative_content = generate_creative_content(prompt, model, tokenizer)

    # Display the output
    print("Generated Content:")
    print("-" * 40)
    print(creative_content)
    print("-" * 40)
