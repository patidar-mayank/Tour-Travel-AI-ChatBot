import streamlit as st
from rag_bot import load_knowledge_base, build_rag_chain

# Load knowledge base
db = load_knowledge_base("data/knowledge.txt")
qa_chain = build_rag_chain(db)

# Streamlit UI
st.set_page_config(page_title="XploreMate Assistant", layout="centered")
st.title("🌍 XploreMate - AI Travel Guide Assistant")
st.markdown("Ask me anything about booking guides, city tours, visa help, and more.")

query = st.text_input("🔍 Enter your question:")

greetings = ["hello", "hi", "hey", "hii", "heyy", "good morning", "good evening"]
domain_keywords = [
    "guide", "travel", "tour", "xploremate", "visa", "hotel",
    "booking", "trip", "city", "location", "places"
]

if query:
    query_lower = query.lower()

    if any(greet in query_lower for greet in greetings):
        st.markdown("👋 Hello! I'm your XploreMate assistant. You can ask me how to book a guide, explore services, or learn about travel help.")

    elif any(keyword in query_lower for keyword in domain_keywords):
        with st.spinner("Thinking... 💭"):
            response = qa_chain.invoke({"query": query})
            answer = response['result']

        st.markdown(f"**🤖 Response:**\n\n{answer}")
        st.markdown("_Want to know more? Ask 'What other features does XploreMate offer?'_")

    else:
        st.error("❌ I'm built only to help with XploreMate-related queries. Please ask something relevant.")

else:
    st.info("💡 Please enter a question to get started.")
