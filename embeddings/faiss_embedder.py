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

import faiss
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from .base_embedder import BaseEmbedder

class FAISSEmbedder(BaseEmbedder):
    def __init__(self, model="text-embedding-ada-002"):
        self.embedder = OpenAIEmbeddings(model=model)
        self.index = None  # Will be initialized after document embedding

    def embed_text(self, text: str):
        return np.array(self.embedder.embed_query(text)).astype("float32")

    def embed_document(self, document_path: str):
        with open(document_path, "r") as file:
            lines = file.readlines()

        embeddings = [self.embed_text(line.strip()) for line in lines]
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings))
        return self.index
