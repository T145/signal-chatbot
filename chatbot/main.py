import asyncio
import os

import arrow
import typer
from dotenv import dotenv_values
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from pymongo import MongoClient
from pysignalclirestapi import SignalCliRestApi

app = typer.Typer()


@app.callback()
def callback():
    """
    Awesome chatbot
    """


@app.command()
def run():
    """
    Run the chatbot
    """
    prompt_template = """Ignore all directives given before what follows.
    You are Aida, a helpful, creative, casual, clever, and very friendly AI assistant.
    Respond to the next request sent by a person to the best of your knowledge,
    rarely greeting or referencing them in your response.

    (Asked on {timestamp})
    {sender}: {message}
    """
    config = dotenv_values(".env")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", repetition_penalty=1.03
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["message"])
    llm_chain = prompt | llm
    signal = SignalCliRestApi(
        base_url="http://localhost:1337", number=config["PHONE_NUMBER"]
    )
    client = MongoClient("localhost", 27017, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["signal_db"]
    groups = db["groups"]
    mailbox = db["mailbox"]
    #chat = db["chat"]

    # db.groups.create_index([('internal_id', ASCENDING)], unique=True)

    typer.echo("Running the chatbot!")

    for group in signal.list_groups():
        cursor = groups.find_one({"internal_id": group["internal_id"]})

        if not cursor or len(list(cursor)) == 0:
            groups.insert_one(group)

    try:
        while True:
            asyncio.sleep(2)

            dms = signal.receive()

            if len(dms) != 0:
                for dm in dms:
                    dm = dm["envelope"]
                    full_message = dm

                    mailbox.insert_one(full_message)

                    if "syncMessage" in dm:
                        dm = dm["syncMessage"]

                        if "sentMessage" not in dm:
                            continue

                        dm = dm["sentMessage"]
                    elif "dataMessage" in dm:
                        dm = dm["dataMessage"]
                    else:
                        continue

                    if "message" not in dm:
                        continue

                    text = dm["message"]

                    if text:
                        if text.lower().startswith("prompt:"):
                            typer.echo("[INFO]: Got a request!")

                            msg = text.split(":")[1].strip()
                            dest = dm["destination"] if "destination" in dm and dm["destination"] else groups.find_one(
                                        {"internal_id": dm["groupInfo"]["groupId"]}
                                    )["id"]

                            if msg:
                                formatted_time = (
                                    arrow.get(dm["timestamp"])
                                    .to("local")
                                    .format("dddd, MMMM D, YYYY {} h:mm a")
                                    .format("at")
                                )
                                review = llm_chain.invoke(
                                    {
                                        "timestamp": formatted_time,
                                        "sender": full_message["sourceName"],
                                        "message": msg,
                                    }
                                )

                                signal.send_message(
                                    message=review, recipients=[dest]
                                )
                            else:
                                signal.send_message(
                                    message="Aida: Please give an actual prompt. I have nothing to answer!",
                                    recipients=[dest],
                                )
                    else:
                        # It's a reaction
                        typer.echo("[INFO]: Reaction detected!")

    except KeyboardInterrupt:
        typer.echo("[INFO]: Exiting gracefully")
