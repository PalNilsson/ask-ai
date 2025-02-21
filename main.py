#!/usr/bin/env python
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors:
# - Paul Nilsson, paul.nilsson@cern.ch, 2025

import argparse
from models.openai_model import OpenAIModel
from embeddings.faiss_embedder import FAISSEmbedder
from utils.query_handler import QueryHandler


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--ai", choices=["openai", "gemini", "deepsearch"], required=True, help="AI model to use")
    parser.add_argument("--doc", required=True, help="Path to the document for retrieval")

    args = parser.parse_args()

    if args.ai == "openai":
        ai_model = OpenAIModel(api_key="YOUR_OPENAI_KEY")
    else:
        raise ValueError("Only OpenAI is implemented. Extend for others.")

    embedder = FAISSEmbedder()
    chatbot = QueryHandler(ai_model, embedder, args.doc)

    print("Chatbot ready! Type your query below:")
    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit"]:
            break
        response = chatbot.get_response(query)
        print("AI:", response)


if __name__ == "__main__":
    main()
