import os
from config import config, PDF_DIR
import streamlit as st
from utils import pdf_loader, load_vector_store, save_vector_store
from chains import qa_


def mode_select() -> str:
    default_value = None
    options = ["handbook pdf VectorStore Enabled", "Pdf Upload Input Enabled"]
    return st.radio("Select Run Type", options, horizontal=True,
                    index=options.index(default_value)
                    if default_value is not None else None)


def mode_select_for_question_input() -> str:
    default_value = None
    question_options = ["List", "Chat"]
    return st.radio("Select Type", question_options, horizontal=True,
                    index=question_options.index(default_value)
                    if default_value is not None else None)


def click_button() -> bool:
    st.session_state.clicked = True


def main():
    """
    application to run private chat bot
    """
    st.set_page_config("PDF ChatBot")
    st.header("PDF Q&A")
    name = mode_select()

    # list to store input user questions
    if 'my_lst' not in st.session_state:
        st.session_state['my_lst'] = []

    if name == "handbook pdf VectorStore Enabled":
        store = load_vector_store()
        if store is not None:
            st.write("Data in Store")
            option = st.radio("Select Run Type", [0, 1, 2, 3], horizontal=True)
            st.write(config.base_config.QUESTIONS[option])
            docs = store.similarity_search(
                config.base_config.QUESTIONS[option], k=60)
            button_clicked = st.button('Generate Answer',
                                       on_click=click_button)
            button_functionality(button_clicked, docs,
                                 config.base_config.QUESTIONS[option])
        else:
            chunks = pdf_loader(os.path.join(PDF_DIR, "handbook.pdf"))
            st.write(chunks)
            save_vector_store(chunks)
            st.write("No Data For handbook.pdf")
    elif name == "Pdf Upload Input Enabled":
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        input_type_name = mode_select_for_question_input()

        # upload pdf embeddings
        if 'store' not in st.session_state:
            st.session_state['store'] = None
        if pdf is not None:
            chunks = pdf_loader(pdf)
            if st.session_state['store'] is None:
                st.session_state['store'] = save_vector_store(chunks, False)

            # chat Interface
            store = st.session_state['store']
            if input_type_name == "List":
                manager()
                user_input = st.session_state['my_lst']
            elif input_type_name == "Chat":
                user_input = st.text_input("Enter Question")
                docs = store.similarity_search(
                    user_input, k=60)

            answer_button_clicked = st.button('Generate Answer',
                                              on_click=click_button)
            if answer_button_clicked and input_type_name == "List":
                if len(user_input) > 0:
                    button_functionality_list(answer_button_clicked, store,
                                              user_input)
                else:
                    st.write("Enter Questions in List")
            elif answer_button_clicked and input_type_name == "Chat":
                button_functionality(answer_button_clicked, docs, user_input)

            if 'my_lst' in st.session_state and st.session_state['my_lst']:
                st.write(st.session_state['my_lst'])

            if st.button('Clear Question List', on_click=click_button):
                st.session_state['my_lst'] = []


def manager():
    with st.expander("User Question List"):
        user_input = st.text_input("Enter Question")
        add_button = st.button("Add", key='add_button')
        if add_button:
            if len(user_input) > 0:
                st.session_state['my_lst'] += [user_input]
            else:
                st.warning("Enter text")


def button_functionality(button_clicked, docs, question):
    if button_clicked:
        context = "\n".join([str(doc.page_content) for doc in docs])
        response = qa_(context, question)
        st.write({
            "Question": question,
            "Answer": response.content
            })

        # Reset the button state after execution
        st.session_state.clicked = False
    else:
        # Set the button state to False if not clicked
        st.session_state.clicked = False


def button_functionality_list(button_clicked, store, questions):
    if button_clicked:
        list_response = []
        for ques in questions:
            docs = store.similarity_search(
                ques, k=60)
            context = "\n".join([str(doc.page_content) for doc in docs])
            response = qa_(context, ques)
            list_response.append({
                "Question": ques,
                "Answer": response.content
                })
        st.write(list_response)
        # Reset the button state after execution
        st.session_state.clicked = False
    else:
        # Set the button state to False if not clicked
        st.session_state.clicked = False


if __name__ == '__main__':
    main()
