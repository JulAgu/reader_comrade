from typing import Literal
from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

mistral_model = "mistral-large-2411"
llm = ChatMistralAI(model=mistral_model, temperature=0)

# Definig DataClasses for deterministic outputs
class AcceptPrompt(BaseModel):
    """ Route a user query to the most relevant datasource. """

    datasource: Literal["answer", "refuse"] = Field(
        ...,
        description="Given a user question choose to answer or refuse it.",
    )

class RouteQuery(BaseModel):
    """ Route a user query to the most relevant datasource. """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
def make_acceptRouter():
    structured_llm_router = llm.with_structured_output(AcceptPrompt)
    # TODO: Verify how to add memory to this
    return
