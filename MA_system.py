from typing import Literal
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import yaml
with open("system_prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

mistral_model = "mistral-large-2411"
llm = ChatMistralAI(model=mistral_model, temperature=0)

# Definig DataClasses for deterministic outputs
class AcceptPrompt(BaseModel):
    """
    Accept or reject a question.
    """
    answ_decision: Literal["answer", "refuse"] = Field(
        ...,
        description="Given a user question choose to answer or refuse it.",
    )
    @validator("answ_decision")
    def validate_datasource(cls, v):
        if v == "answer" or v == "refuse":
            return v
        else:
            raise ValueError("datasource must be either 'answer' or 'refuse'")


class RouteQuery(BaseModel):
    """ Route a user query to the most relevant datasource. """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    @validator("datasource")
    def validate_datasource(cls, v):
        if v == "vectorstore" or v == "websearch":
            return v
        else:
            raise ValueError("datasource must be either 'vectorstore' or 'websearch'")

def make_acceptRouter():
    structured_llm_router = llm.with_structured_output(AcceptPrompt)
    # TODO: Verify how to add memory to this
    return
