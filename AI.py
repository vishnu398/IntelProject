# Import required libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Function to generate response
def generate_response(prompt, max_length=50):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Generate response using the model
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        temperature=0.9,   # Adjust temperature for diversity
        top_p=0.9,         # Adjust top_p for nucleus sampling
        num_beams=5,       # Use beam search for better responses
        early_stopping=True,
        attention_mask=input_ids.ne(tokenizer.pad_token_id).long()  # Ensure attention mask is set
    )
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Chat function
def chat():
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input)
        print(f"Bot: {response}\n")

# Run the chatbot
chat()
