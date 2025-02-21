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

from embeddings.faiss_embedder import FAISSEmbedder
from models.openai_model import OpenAIModel  # Changeable based on AI used

class QueryHandler:
    def __init__(self, ai_model, embedder, document_path):
        self.model = ai_model
        self.embedder = embedder
        self.document_path = document_path
        self.index = self.embedder.embed_document(document_path)

    def retrieve_relevant_text(self, query):
        query_embedding = self.embedder.embed_text(query)
        _, indices = self.index.search(query_embedding.reshape(1, -1), k=3)

        with open(self.document_path, "r") as file:
            lines = file.readlines()

        retrieved_text = " ".join([lines[i].strip() for i in indices[0]])
        return retrieved_text

    def get_response(self, query):
        context = self.retrieve_relevant_text(query)
        return self.model.generate_response(f"Context: {context}\nQuestion: {query}")
