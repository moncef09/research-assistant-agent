"""
Streamlit app for interacting with the research assistant agent.

To run this app, execute:

```bash
streamlit run app.py
```

You must have the environment variables required by `agent_flow` set (see README.md).
"""

import streamlit as st

from agent_flow import build_agent


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


def main():
    st.set_page_config(page_title="Research Assistant Agent", page_icon="📚")
    st.title("🔍 Research Assistant Agent")
    st.markdown(
        "Ask a question about the research papers in the database and the agent will fetch relevant documents, "
        "summarise them and provide a beginner‑friendly explanation."
    )

    question = st.text_input("Your question", placeholder="e.g. Explain the volatility of Bitcoin")
    if st.button("Submit"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                agent = get_agent()
                response = agent.invoke({"question": question})
            st.success("Done!")
            st.write(response)


if __name__ == "__main__":
    main()
