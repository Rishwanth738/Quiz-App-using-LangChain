import dotenv
dotenv.load_dotenv()
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


llm = ChatOpenAI()
output_parser = StrOutputParser()

classifier_template = PromptTemplate.from_template(
    "You are given a topic by the user and now you have to ask them a question about that topic. Do not ask them a question that involves their opinion. The question must be factual and related to the topic. For example, if the topic is 'Pokemon', you can ask 'What is Pikachu's type?'. **DO NOT** ask them a question like 'What do you think was the best generation of Pokemon?'. "
    "Avoid repeating any of these previous questions: {asked_questions}\n\nTopic - {topic}\nQuestion:"
)

question_template = PromptTemplate.from_template(
    "The user was asked a question and they answered it. Now check whether the answer is correct or not. "
    "Just that, don't give them a full explanation.\nQuestion - {question}\nAnswer - {answer}\nResponse:"
)

correct_template = PromptTemplate.from_template(
    "The user was asked the following question and they answered it correctly. Now ask a more difficult question.Do not ask them a question that involves their opinion. The question must be factual and related to the topic. For example, if the topic is 'Pokemon', you can ask 'What is Pikachu's type?'. **DO NOT** ask them a question like 'What do you think was the best generation of Pokemon?'."
    "on the same topic. Avoid repeating any of these: {asked_questions}\n\nPrevious Question: {question}\nNew Question:"
)

incorrect_template = PromptTemplate.from_template(
    "The user was asked the following question and they answered it incorrectly. Give the correct answer and explain it clearly. "
    "Then, ask them a simpler question related to the same topic.Do not ask them a question that involves their opinion. The question must be factual and related to the topic. For example, if the topic is 'Pokemon', you can ask 'What is Pikachu's type?'. **DO NOT** ask them a question like 'What do you think was the best generation of Pokemon?'. Avoid repeating any of these: {asked_questions}\n\n"
    "Question: {question}\nCorrect Answer:\nExplanation:"
)

classifier_chain = classifier_template | llm | output_parser
question_chain = question_template | llm | output_parser
correct_chain = correct_template | llm | output_parser
incorrect_chain = incorrect_template | llm | output_parser

def build_pipeline(user_answer):
    passthrough = RunnablePassthrough.assign(answer=lambda _: user_answer)

    parallel_input = RunnableParallel(
        question=RunnablePassthrough(),
        answer=RunnablePassthrough()
    )

    def router(result):
        res = result.lower()
        st.session_state["last_eval"] = result.strip()

        if res.strip() == "correct":
            return correct_chain
        elif res.strip() == "incorrect":
            return incorrect_chain
        elif "correct" in res and "incorrect" not in res:
            return correct_chain
        elif "incorrect" in res:
            return incorrect_chain
        else:
            class DummyChain:
                def invoke(self, input_dict):
                    return result.strip()
            return DummyChain()

    class Pipeline:
        def invoke(self, input_dict):
            eval_result = question_chain.invoke(input_dict)
            next_chain = router(eval_result)
            if next_chain is correct_chain:
                st.session_state.score += 1
            return next_chain.invoke(input_dict)

    return Pipeline()

st.title("Smart Quiz App (LLM-powered)")
st.markdown("Test yourself on any topic. We'll remember previous questions so no repeats!")

if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.asked_questions = []
    st.session_state.question = ""
    st.session_state.topic = ""
    st.session_state.score = 0

if not st.session_state.started:
    st.session_state.topic = st.text_input("Enter a topic to start the quiz:")
    if st.button("Start Quiz"):
        asked_qs_str = ", ".join(st.session_state.asked_questions)
        st.session_state.question = classifier_chain.invoke({
            "topic": st.session_state.topic,
            "asked_questions": asked_qs_str
        })
        st.session_state.asked_questions.append(st.session_state.question)
        st.session_state.started = True
        st.session_state.score = 0 
        st.rerun()
else:
    st.subheader("Question:")
    st.markdown(f"**{st.session_state.question}**")
    st.markdown(f"**Score:** {st.session_state.score}")
    user_answer = st.text_input("💬 Your Answer (type 'exit' to end):")

    if st.button("Submit"):
        if user_answer.lower().strip() == "exit":
            st.success("Quiz ended. Hope you learned something cool!")
            st.session_state.started = False
            st.session_state.asked_questions.clear()
            st.session_state.question = ""
            st.session_state.score = 0
            st.rerun()
        else:
            pipeline = build_pipeline(user_answer)
            next_q_or_response = pipeline.invoke({
                "question": st.session_state.question,
                "answer": user_answer,
                "asked_questions": ", ".join(st.session_state.asked_questions)
            })

            
            if "last_eval" in st.session_state and "incorrect" in st.session_state["last_eval"].lower():
                if "New Question:" in next_q_or_response:
                    parts = next_q_or_response.split("New Question:")
                    st.session_state.question = parts[-1].strip()
                else:
                    st.session_state.question = next_q_or_response.strip()
            else:
                st.session_state.question = next_q_or_response.strip()

            st.session_state.asked_questions.append(st.session_state.question)
            st.rerun()
