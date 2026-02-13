from app.rag import answer

question = "How much compensation is given if a worker dies in an accident?"

print("\nQuestion:", question)
print("\nAnswer:\n")

response = answer(question)
print(response)
