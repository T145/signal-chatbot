import sys
sys.dont_write_bytecode = True

import asyncio
import uuid
import json
from chatbot.helpers import (
    State,
    RequestAssistance,
    AsyncMongoDBSaver
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessageChunk,
    RemoveMessage
)
from os import environ
from dotenv import dotenv_values
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple, Annotated, Literal, List, Union
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
#from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from duckduckgo_search import AsyncDDGS
from langchain_community.tools import WikipediaQueryRun, BaseTool, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from datasets import load_dataset
from langchain_core.tools import StructuredTool
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
# from langchain_core.utils import stringify_dict
# import weaviate
# from langchain.text_splitter import RecursiveJsonSplitter
# from langchain_community.document_loaders import JSONLoader
# from langchain_weaviate.vectorstores import WeaviateVectorStore
# import weaviate.classes.config as wvcc
# from langchain_community.document_loaders.base import BaseLoader


config = dotenv_values('.env')
#environ['TAVILY_API_KEY'] = config['TAVILY_API_KEY']
environ['HUGGINGFACEHUB_API_KEY'] = config['HUGGINGFACEHUB_API_KEY']


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> ToolNode:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# async def _aget_results(query):
#     return await AsyncDDGS(proxy=None).atext(query, max_results=5, safesearch='off', region='us-en')


# @tool('duckduckgo')
# async def search_duckduckgo(
#     queries: Annotated[str, 'Query for the search engine.']
# ):
#     """A search engine optimized for comprehensive, accurate, and trusted results.
#     Useful for when you need to answer questions about current & modern events."""
#     tasks = [_aget_results(q) for q in [queries]]
#     results = await asyncio.gather(*tasks)
#     return results


# TODO: Use googlesearch https://github.com/Nv7-GitHub/googlesearch
# Why async? It allows streaming and removes all token limitations
async def run():
    async with AsyncMongoDBSaver.from_conn_info(
        host='localhost', port=27017, db_name='checkpoints'
    ) as checkpointer:
        tools = [
            #TavilySearchResults(max_results=2),
            #recall,
            #search_duckduckgo
            #WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2500)),
            #RequestAssistance
        ]
        model = ChatOllama(
            model='llama3.1:8b',
            mirostat=2,
            keep_alive=-1, # https://github.com/ollama/ollama/issues/4427#issuecomment-2143525092
            streaming=True
        ).bind_tools(
            tools,
            #tool_choice='any' # Llama loves tool calls, often above its own reasoning, so we force it to always call a tool and have one that makes it rely on itself.
        )
        in_memory_store = InMemoryStore()

        # signal_db = checkpointer.client['signal']
        # groups = signal_db['groups']
        # await groups.insert_one({"name": "John"})


        async def call_model(state: State, config: RunnableConfig, *, store: BaseStore):
            user_id = config['configurable']['user_id']
            namespace = ('memories', user_id)
            memories = store.search(namespace)
            #info = "\n".join([d.value["data"] for d in memories])
            last_message = state['messages'][-1]

            if isinstance(last_message, HumanMessage):
                memory = last_message.content.strip()
                store.put(namespace, str(uuid.uuid4()), {'data': memory})

            response = await model.ainvoke(state['messages'])
            return {'messages': [response]}


        workflow = StateGraph(State)

        workflow.add_node('agent', call_model)
        workflow.add_node('tools', create_tool_node_with_fallback(tools))
        workflow.add_conditional_edges('agent', tools_condition)
        workflow.add_edge('tools', 'agent')
        workflow.set_entry_point('agent')

        app = workflow.compile(
            checkpointer=checkpointer,
            store=in_memory_store
        )

        with open('graph.png', 'wb') as png:
            png.write(app.get_graph().draw_mermaid_png())

        config = {'configurable': {'thread_id': '2', 'user_id': '2'}}
        latest_checkpoint = await checkpointer.aget(config)
        inputs = {'messages': []}

        if not latest_checkpoint:
            system_prompt = """Ignore all directives given before what follows.
            Your name is Aida.

            On Aida
            - You are helpful, inquisitive, creative, casual, clever, and very friendly.
            - Occasionally greet or reference the users in your response.
            - You have a photographic memory.

            On Responses
            - Preface responses with your name followed by a colon character.
            - Because you're casual, you can use a wide variety of emoticons.
            - Paraphrase responses to avoid being repetitive, but keep any relevant information.
            - Limit emoji usage to a couple times maximum.
            - Using emoticons is not necessary.
            - Avoid using the phrase 'ahah'.

            On Tools
            - Do not mention that you use tools or used a tool.
            - If you already know the answer, use the "recall" tool.
            - Do not apologize when tools fail: if you still don't know, ask the user for help.
            - Do not include tool prompts in the response.
            - Do not manufacture user fake user input to include in the response."""
            # system_prompt = """Ignore all directives given before what follows.
            # You are Aida: helpful, creative, casual, clever, and very friendly.
            # You have a photographic memory.
            # Respond to requests sent by users to the best of your knowledge,
            # and occasionally greet or reference the users in your response.
            # Preface responses with your name followed by a colon character.
            # Because you're casual, feel free to paraphrase and use emoticons!
            # Avoid using the phrase 'ahah'."""

            inputs['messages'].append(SystemMessage(content=system_prompt))

        inputs['messages'].append(HumanMessage(content="Taylor: Who was Eratosthenes?"))

        async for output in app.astream(inputs, config, stream_mode='updates'):
            # stream_mode='updates' yields dictionaries with output keyed by node name
            for key, value in output.items():
                value['messages'][-1].pretty_print()


if __name__ == '__main__':
    asyncio.run(run())
