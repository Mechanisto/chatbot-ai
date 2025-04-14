from response import generate_response

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = generate_response(user_input)
    print("Chatbot:", response)
