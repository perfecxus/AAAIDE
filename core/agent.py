from google.adk.agents import LlmAgent
from google.adk.agents.llm_agent import InstructionProvider
from typing_extensions import TypeAlias
from typing import Union
from core.tools.data_discovery_tool import DataDiscoveryTool

from .prompts import return_client_partner_prompt, return_requirement_gathering_prompt


class AAIDEAgent(LlmAgent):
    domain: Union[str,InstructionProvider] = 'generic domain'

    def return_agent_instructions(self,func,domain):
        return str(func()) + str(domain)


class ClientPartnerLlmAgent(AAIDEAgent):
    instruction: Union[str,InstructionProvider] = super().return_agent_instructions(return_client_partner_prompt,super().domain)
    #return_client_partner_prompt(super().domain)

ClientPartnerAgent: TypeAlias = ClientPartnerLlmAgent

class RequirementLlmAgent(AAIDEAgent):
    instruction: Union[str,InstructionProvider] = super().return_agent_instructions(return_requirement_gathering_prompt,super().domain)

RequirementGatheringAgent: TypeAlias = RequirementLlmAgent

class DataDiscoveryFWKAgent(AAIDEAgent):
    instruction: Union[str,InstructionProvider] = super().return_agent_instructions(return_requirement_gathering_prompt,super().domain)
    data_discovery_tool = DataDiscoveryTool()
    tools = [data_discovery_tool]

DataDiscoveryAgent: TypeAlias = DataDiscoveryFWKAgent
