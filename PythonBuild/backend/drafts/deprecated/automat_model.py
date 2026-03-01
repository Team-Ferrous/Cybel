
from pydantic     import BaseModel, ConfigDict
from langchain_core.prompts.base import BasePromptTemplate

class AutomatModel(BaseModel):
    prompt: BasePromptTemplate
    model_config = ConfigDict(arbitrary_types_allowed=True)
