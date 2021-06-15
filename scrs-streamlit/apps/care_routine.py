import streamlit as st
from .routine_chatbot import *
def care_routine():
    new_choice = st.selectbox("Frequently Ask Questions: ",
    ('Choose one',
    'What are Skin Diseases?',
    'What is the effect of Melanoma?',
    'What is the effect of Insect Bite on skin?',
    'Is acne temperary disease?',
    'What causes acne?',
    'Is Body Temperature increase with skin diseases?',
    'What is the disorders of skin pigment?',
    'How much Basal Cell effect?',
    'In which Age Actinic grow?',
    'others'
    ))

    if new_choice == 'What is the effect of Melanoma?':
        st.write('Response:')
        st.success("""
        Melanoma can spread to parts of your body far away
        from where the cancer started. This is called advanced,
        metastatic, or stage IV melanoma.
        It can move to your lungs, liver, brain, bones, digestive system, and lymph nodes.
        """)

    elif new_choice == 'What are Skin Diseases?':
        st.write('Response:')
        st.success("""
            Skin diseases are a broad range of conditions affecting
            the skin, and include diseases caused by bacterial
            infections, viral infections, fungal infections, allergic
            reactions, skin cancers, and parasites.
        """)
    elif new_choice == 'What is the effect of Insect Bite on skin?':
        st.write('Response:')
        st.success("""
        Your body's immediate response will include redness and
        swelling at the site of the bite or sting.
        """)

    elif new_choice == 'Is acne temperary disease?':
        st.write('Response:')
        st.success("""
        Most acne's pimples will eventually clear up on their own.
        But if your pimple is very large or painful. You can
        check severity level here. Just go to Analyse tab.
        """)
    elif new_choice == 'What causes acne?':
        st.write('Response:')
        st.success("""
        Many factors contribute to the development of acne, but recent research
        shows that diet can play a significant role in acne development.
        Certain foods raise your blood sugar more quickly than others.
        """)
    elif new_choice == 'What is the disorders of skin pigment?':
        st.write('Response:')
        st.success("""
            Vitiligo is a condition in which the skin loses its pigment cells (melanocytes).
            This can result in discolored patches in different areas of the body, including the skin,
            hair and mucous membranes. Vitiligo (vit-ih-LIE-go) is a disease that causes loss of skin
            color in patches.
        """)
    elif new_choice == 'How much Basal Cell effect?':
        st.write('Response:')
        st.success("""
            Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma
            begins in the basal cells — a type of cell
            within the skin that produces new skin cells as old ones die off.
        """)

    elif new_choice == 'others':
        st.write('Response:')
        user_text = st.text_input("Enter Query Here:")
        # st.markdown('<br>', unsafe_allow_html=True)
        click = st.button("Send")
        if click:
            col1, col2 = st.beta_columns((2,4))
            col3, col4 = st.beta_columns((4,2))
            if user_text != "":
                col2.success(user_text)
                response = text_send(user_text)
                col3.success(response)
