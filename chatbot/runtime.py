import json
import pprint
import time
import os
import arrow
from pysignalclirestapi import SignalCliRestApi
from dotenv import dotenv_values
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


def main():
    prompt_template = """
    Ignore all directives given before what follows.
    You are Aida, a helpful, creative, casual, clever, and very friendly AI assistant.
    Respond to the next request sent by a person to the best of your knowledge,
    rarely greeting or referencing them in your response.

    (Asked on {timestamp})
    {sender}: {message}
    """
    config = dotenv_values('.env')
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['HUGGINGFACEHUB_API_TOKEN']
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        repetition_penalty=1.03
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["message"]
    )
    llm_chain = prompt | llm
    signal = SignalCliRestApi(
        base_url = 'http://localhost:8080',
        number = config['PHONE_NUMBER']
    )
    client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['signal_db']
    groups = db['groups']
    mailbox = db['mailbox']
    chat = db['chat']

    #db.groups.create_index([('internal_id', ASCENDING)], unique=True)

    print('Running the chatbot!')

    for group in signal.list_groups():
        cursor = groups.find_one({'internal_id': group['internal_id']})

        if (not cursor or len(list(cursor)) == 0):
            groups.insert_one(group)

    try:
        while True:
            time.sleep(2)

            dms = signal.receive()

            if len(dms) != 0:
                for dm in dms:
                    dm = dm['envelope']
                    full_message = dm

                    mailbox.insert_one(full_message)

                    if 'syncMessage' in dm:
                        dm = dm['syncMessage']

                        if 'sentMessage' not in dm:
                            continue

                        dm = dm['sentMessage']
                    elif 'dataMessage' in dm:
                        dm = dm['dataMessage']
                    else:
                        continue

                    if 'message' not in dm:
                        continue

                    text = dm['message']

                    if text:
                        if text.lower().startswith('prompt:'):
                            print('[INFO]: Got a request!')

                            def get_dest():
                                if 'destination' in dm and dm['destination']:
                                    return dm['destination']
                                else:
                                    return groups.find_one({'internal_id': dm['groupInfo']['groupId']})['id']

                            msg = text.split(':')[1].strip()

                            if msg:
                                formatted_time = arrow.get(dm['timestamp']).to('local').format('dddd, MMMM D, YYYY {} h:mm a').format('at')
                                review = llm_chain.invoke({
                                    "timestamp": formatted_time,
                                    "sender": full_message['sourceName'],
                                    "message": msg
                                })

                                signal.send_message(message=review, recipients=[get_dest()])
                            else:
                                signal.send_message(message='Aida: Please give an actual prompt. I have nothing to answer!', recipients=[get_dest()])
                    else:
                        # It's a reaction
                        print('[INFO]: Reaction detected!')

    except KeyboardInterrupt:
        print('Exiting gracefully')
    finally:
        client.close()

    if __name__ == '__main__':
        main()
