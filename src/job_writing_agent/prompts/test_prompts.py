from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


REVISION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an expert job application writer. Revise the draft based on BOTH the self-evaluation and external feedback provided."
        ),
        HumanMessagePromptTemplate.from_template(
            """
    # Original Draft Content with Evaluation Section at the end
    {draft}

    # Candidates' Feedback (Human Feedback)
    {feedback}

    # Critique Feedback (AI Feedback)
    {critique_feedback}
    
    Based on the self evaluation in the Original Draft, Critique Feedback and the Candidates' Feedback, revise the content taking essence of the self evaluation, Critique Feedback and the Candidates' Feedback into account. Do not repeat the same content from the Original Draft, Critique Feedback and the Candidates' Feedback.

    Return the content of the revised draft. Make sure the output is only the content that is the revised content and nothing else.
    """
        ),
    ]
)

print(
    REVISION_PROMPT.format_messages(
        draft="Hello, how are you?",
        feedback="I like your draft.",
        critique_feedback="Your draft is good.",
    )
)
