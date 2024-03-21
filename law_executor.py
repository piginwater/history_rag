import argparse
import logging
import sys
import re
import os
import argparse

import requests
from pathlib import Path
from urllib.parse import urlparse

from llama_index import ServiceContext, StorageContext
from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms import OpenAI
from llama_index.readers.file.flat_reader import FlatReader
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser.text import SentenceWindowNodeParser

from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole, PromptTemplate
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor import SentenceTransformerRerank
# from llama_index.indices import ZillizCloudPipelineIndex
from custom.zilliz.base import ZillizCloudPipelineIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import BaseNode, ImageNode, MetadataMode

from custom.law_sentence_window import LawSentenceWindowNodeParser
from custom.llms.QwenLLM import QwenUnofficial
from custom.llms.GeminiLLM import Gemini
from custom.llms.proxy_model import ProxyModel
from pymilvus import MilvusClient

QA_SYSTEM_PROMPT = "你是一个严谨的中国法律学的专家，你会仔细阅读包含法律条文的资料并给出准确的回答,你的回答都会非常准确"
QA_PROMPT_TMPL_STR = (
    "请你仔细阅读给出的法律法规资料进行回答，如果发现资料无法得到答案，就回答不知道。 \n"
    "仅根据以下搜索出的相关法律法规资料，而不依赖于先验知识，回答问题。\n"
    "每个法律法规资料，都有三个附加信息，1. 该资料所属法律法规；2. 该资料在法律法规中所属的章节；3. 该资料在法律法规中所属的章节中所属的条款。回答法律法规条款内容时，请参考\n"
    "搜索的相关法律法规资料如下所示。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "问题: {query_str}\n"
    "答案: "
)
REFINE_PROMPT_TMPL_STR = (
    "你是一个严谨的中国法律学的专家，你严格按以下方式工作"
    "1.只有原答案为不知道时才进行修正,否则输出原答案的内容。\n"
    "2.如果感到疑惑的时候，就用原答案的内容回答。\n"
    "新的相关法律法规资料: {context_msg}\n"
    "原问题: {query_str}\n"
    "原答案: {existing_answer}\n"
    "新答案: "
)


class Executor:
    def __init__(self, model):
        pass

    def build_index(self, path, overwrite):
        pass

    def build_query_engine(self):
        pass

    def delete_file(self, path):
        pass

    def query(self, question):
        pass


class LawExecutor(Executor):
    def __init__(self, config):
        self.index = None
        self.query_engine = None
        self.config = config
        self.node_parser = LawSentenceWindowNodeParser.from_defaults(
            sentence_splitter=lambda text: re.findall(
                "[^,.;。？！]+[,.;。？！]?", text),
            window_size=config.milvus.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",)

        embed_model = HuggingFaceEmbedding(model_name=config.embedding.name)

        # 使用Qwen 通义千问模型
        if config.llm.name.find("qwen") != -1:
            llm = QwenUnofficial(
                temperature=config.llm.temperature, model=config.llm.name, max_tokens=2048)
        elif config.llm.name.find("gemini") != -1:
            llm = Gemini(temperature=config.llm.temperature,
                         model_name=config.llm.name, max_tokens=2048)
        elif 'proxy_model' in config.llm:
            llm = ProxyModel(model_name=config.llm.name, api_base=config.llm.api_base, api_key=config.llm.api_key,
                             temperature=config.llm.temperature,  max_tokens=2048)
            print(
                f"使用{config.llm.name},PROXY_SERVER_URL为{config.llm.api_base},PROXY_API_KEY为{config.llm.api_key}")
        else:
            api_base = None
            if 'api_base' in config.llm:
                api_base = config.llm.api_base
            llm = OpenAI(api_base=api_base, temperature=config.llm.temperature,
                         model=config.llm.name, max_tokens=2048)

        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model)
        set_global_service_context(service_context)
        rerank_k = config.milvus.rerank_topk
        self.rerank_postprocessor = SentenceTransformerRerank(
            model=config.rerank.name, top_n=rerank_k)
        self._milvus_client = None
        self._debug = False

    def set_debug(self, mode):
        self._debug = mode

    def build_index(self, path, overwrite):
        config = self.config
        vector_store = MilvusVectorStore(
            uri=f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name=config.milvus.collection_name,
            overwrite=overwrite,
            dim=config.embedding.dim)
        self._milvus_client = vector_store.milvusclient

        if path.endswith('.txt'):
            if os.path.exists(path) is False:
                print(f'(rag) 没有找到文件{path}')
                return
            else:
                documents = FlatReader().load_data(Path(path))
        elif os.path.isfile(path):
            print('(rag) 目前仅支持txt文件')
        elif os.path.isdir(path):
            if os.path.exists(path) is False:
                print(f'(rag) 没有找到目录{path}')
                return
            else:
                documents = SimpleDirectoryReader(path).load_data()
        else:
            return

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(
            nodes, storage_context=storage_context, show_progress=True)

    def _get_index(self):
        config = self.config
        vector_store = MilvusVectorStore(
            uri=f"http://{config.milvus.host}:{config.milvus.port}",
            collection_name=config.milvus.collection_name,
            dim=config.embedding.dim)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store)
        self._milvus_client = vector_store.milvusclient

    def build_query_engine(self):
        config = self.config
        if self.index is None:
            self._get_index()
        self.query_engine = self.index.as_query_engine(similarity_top_k=config.milvus.retrieve_topk, node_postprocessors=[
            self.rerank_postprocessor,
            MetadataReplacementPostProcessor(target_metadata_key="window")])

        message_templates = [
            ChatMessage(content=QA_SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=QA_PROMPT_TMPL_STR,
                role=MessageRole.USER,
            ),
        ]
        chat_template = ChatPromptTemplate(message_templates=message_templates)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": chat_template,
                "response_synthesizer:refine_template": PromptTemplate(REFINE_PROMPT_TMPL_STR)}
        )

    def delete_file(self, path):
        config = self.config
        if self._milvus_client is None:
            self._get_index()
        num_entities_prev = self._milvus_client.query(
            collection_name='history_rag', filter="", output_fields=["count(*)"])[0]["count(*)"]
        res = self._milvus_client.delete(
            collection_name=config.milvus.collection_name, filter=f"file_name=='{path}'")
        num_entities = self._milvus_client.query(
            collection_name='history_rag', filter="", output_fields=["count(*)"])[0]["count(*)"]
        print(
            f'(rag) 现有{num_entities}条，删除{num_entities_prev - num_entities}条数据')

    def query(self, question):
        if self.index is None:
            self._get_index()
        if question.endswith('?') or question.endswith('？'):
            question = question[:-1]
        if self._debug is True:
            contexts = self.query_engine.retrieve(QueryBundle(question))
            for i, context in enumerate(contexts):
                print(f'{question}', i)
                content = context.node.get_content(
                    metadata_mode=MetadataMode.LLM)
                print(content)
            print('-------------------------------------------------------参考资料---------------------------------------------------------')
        response = self.query_engine.query(question)
        return response
