from transformers import pipeline

knowledge_base = """
Einstein's theory of relativity, comprised of special and general relativity, revolutionized our understanding of space, time, gravity, 
and the universe. Special relativity, published in 1905, established that the laws of physics are the same for all observers in uniform 
motion, and the speed of light is constant for all observers. This led to concepts like time dilation and length contraction, where time 
and space are relative to the observer's motion. General relativity, published in 1915, extended these ideas to include gravity, describing 
it as the curvature of spacetime caused by mass and energy. This theory explains phenomena like the bending of light around massive objects
 and the existence of black holes. Together, these theories demonstrate that space and time are intertwined as spacetime, and that gravity
  is not just a force but a manifestation of spacetime curvature. 
"""
qa_pipeline = pipeline('question-answering', model="distilbert-base-cased-distilled-squad")


def anstheque(question, context=knowledge_base):
    response = qa_pipeline(question=question, context=context)
    answer = response['answer']
    score = response['score']
    return answer, score


def main():
    print("Welcome to the Question answering bot")
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Goodbye")
            break
        answer, confidence = anstheque(question)
        print(f"Answer: {answer} (Confidence: {confidence:.2f})")
main()