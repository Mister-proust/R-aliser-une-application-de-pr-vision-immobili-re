import os
import sys
from dotenv import load_dotenv


sys.path.extend([os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))])

from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent

from agentia.estimation_tool import estimate_property

class RealEstateAgent:
    def __init__(self, model_name: str = "mistral-large-latest", temperature: float = 0.7):
        load_dotenv()
        
        self.model = ChatMistralAI(
            model=model_name,
            temperature=temperature,
        )

        self.system_prompt = (
            """
            Tu es un expert immobilier français.
            Tu aides les utilisateurs à estimer le prix de leurs biens immobiliers
            en utilisant l'outil 'estimate_property'.
            Réponds en français.
            """
        )

        self.agent = create_agent(
            model=self.model,
            tools=[estimate_property],
            system_prompt=self.system_prompt
        )

    def ask(self, query: str):
        response = self.agent.invoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

if __name__ == "__main__":
    print("--- Initialisation de l'Agent Immobilier ---")
    agent = RealEstateAgent()
    
    question = input("Pose ta question à l'agent immobilier : ")
    print(f"\nUtilisateur : {question}")
    
    try:
        reponse = agent.ask(question)
        print(f"\nAgent : {reponse}")
    except Exception as e:
        print(f"\nErreur : {e}")
