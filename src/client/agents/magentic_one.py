import anthropic
import os
from ..agent import AgentClient
from copy import deepcopy
from typing import List
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console


class MagenticAgent(AgentClient):
    def __init__(self, api_args=None, *args, **config):

        super().__init__(*args, **config)

        if not api_args:
            api_args = {}
        api_args = deepcopy(api_args)
        self.key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')

        client = OpenAIChatCompletionClient(model="gpt-4o")  # can be added as a param?
        self.agent = MagenticOne(client=client)
        api_args["model"] = api_args.pop("model", None)
        if not self.key:
            raise ValueError("OPENAI API KEY is required, please assign api_args.key or set OPENAI_API_KEY "
                             "environment variable.")
        
        if not api_args["model"]:
            raise ValueError("Magentic One model requires OPENAI_API_KEY, please assign api_args.model.")
        
        self.api_args = api_args
        if not self.api_args.get("stop_sequences"):
            self.api_args["stop_sequences"] = [anthropic.HUMAN_PROMPT]

    async def inference(self, history: List[dict]) -> str:
            
        prompt = ""
        for message in history:
            if message["role"] == "user":
                prompt += anthropic.HUMAN_PROMPT + message["content"]
            else:
                prompt += anthropic.AI_PROMPT + message["content"]

        prompt += anthropic.AI_PROMPT

        result = await Console(self.agent.run_stream(task=prompt))
        # result = self.agent.run_stream(task=prompt)

        return result
