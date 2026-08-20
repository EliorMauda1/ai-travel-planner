"""Phase 0: minimal end-to-end LangChain -> OpenAI wiring check."""

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = PromptTemplate.from_template(
    "Suggest 3 interesting things to do in {destination} for a first-time visitor."
)

model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model


def main() -> None:
    response = chain.invoke({"destination": "Kyoto, Japan"})
    print(response.content)


if __name__ == "__main__":
    main()
