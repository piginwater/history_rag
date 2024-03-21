"""Customize Simple node parser."""
from typing import Any, Callable, List, Optional, Sequence
from bisect import bisect_right

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import build_nodes_from_splits
from llama_index.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.schema import BaseNode, Document, MetadataMode
from llama_index.utils import get_tqdm_iterable
import re

DEFAULT_WINDOW_SIZE = 3
DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"

meta_key_tilte="所属法律法规"
meta_key_chunck_title="所属章节"
meta_key_node_title="所属条款"

class LawSentenceWindowNodeParser(NodeParser):
    # sentence_splitter: Callable[[str], List[str]] = Field(
    #     default_factory=split_by_sentence_tokenizer,
    #     description="The text splitter to use when splitting documents.",
    #     exclude=True,
    # )
    window_size: int = Field(
        default=DEFAULT_WINDOW_SIZE,
        description="The number of sentences on each side of a sentence to capture.",
        gt=0,
    )
    window_metadata_key: str = Field(
        default=DEFAULT_WINDOW_METADATA_KEY,
        description="The metadata key to store the sentence window under.",
    )
    original_text_metadata_key: str = Field(
        default=DEFAULT_OG_TEXT_METADATA_KEY,
        description="The metadata key to store the original sentence in.",
    )
    split_line: str = Field(
        default="---",
        description="文本的分隔符",
    )

    @classmethod
    def class_name(cls) -> str:
        return "LawSentenceWindowNodeParser"


    @classmethod
    def from_defaults(
        cls,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        window_metadata_key: str = DEFAULT_WINDOW_METADATA_KEY,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "LawSentenceWindowNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        return cls(
            sentence_splitter=sentence_splitter,
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            nodes = self.build_window_nodes_from_documents([node])
            all_nodes.extend(nodes)

        return all_nodes

    def __remove_whitespace(self,text):
        if text is None or len(text) == 0:
            return text
        # 使用正则表达式去除两头的回车、换行、半角空格和全角空格
        cleaned_text = re.sub(r'^[\r\n\s\u0020\u3000]*|[\r\n\s\u0020\u3000]*$', '', text)
        return cleaned_text
    def __get_first_line(self,text):
        # 使用 splitlines() 方法获取文本的各行内容，并返回第一行
        lines = text.splitlines()
        if lines:
            return lines[0]
        else:
            return None
    def is_chapter_chunk(self,title):
        '判断当前的块是否为章节块'
        match = re.search(r"第(.*?)章", title)
        if match is not None:
            return True
        return False
    def extract_item(self,text):
        match = re.search(r"第(.*?)条[\s\u3000]", text)
        if match:
            return match.group()
        else:
            return None
    def analyze_titles(self, text):
        return self.__get_first_line(text)

    def build_window_nodes_from_documents(
        self, documents: Sequence[Document]
    ) -> List[BaseNode]:
        """Build window nodes from documents."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = self.__remove_whitespace(doc.text)
            title = self.analyze_titles(text)
            doc.metadata[meta_key_tilte] = title
            nodes = []
            for chunk in  text.split(self.split_line):
                chunk = self.__remove_whitespace(chunk)
                current_chunk_title = self.__remove_whitespace(self.__get_first_line(chunk))
                current_node_title=""

                # 非章节块，直接将chunk作为内容
                if not self.is_chapter_chunk(current_chunk_title):
                    line_nodes = build_nodes_from_splits(
                        [chunk],
                        doc,
                        id_func=self.id_func,
                    )
                    line_nodes[0].metadata[meta_key_tilte] = title
                    line_nodes[0].metadata[meta_key_chunck_title] = current_chunk_title
                    line_nodes[0].metadata[meta_key_node_title] = ""
                    nodes.extend(line_nodes)
                    continue
                for sent in chunk.splitlines():
                    sent = self.__remove_whitespace(sent)
                    if len(sent) == 0:
                        continue
                    item = self.__remove_whitespace(self.extract_item(sent))
                    if item is not None:
                        current_node_title = item

                    line_nodes = build_nodes_from_splits(
                        [sent],
                        doc,
                        id_func=self.id_func,
                    )
                    line_nodes[0].metadata[meta_key_tilte] = title
                    line_nodes[0].metadata[meta_key_chunck_title] = current_chunk_title
                    line_nodes[0].metadata[meta_key_node_title] = current_node_title
                    nodes.extend(line_nodes)
            for i, node in enumerate(nodes):
                window_nodes = nodes[
                    max(0, i - self.window_size) : min(i + self.window_size, len(nodes))
                ]

                node.metadata[self.window_metadata_key] = " ".join(
                    [n.text for n in window_nodes]
                )
                node.metadata[self.original_text_metadata_key] = node.text

                # exclude window metadata from embed and llm
                node.excluded_embed_metadata_keys.extend(
                    [self.window_metadata_key, self.original_text_metadata_key, 'file_path', 'file_name', 'filename', 'extension']
                )
                node.excluded_llm_metadata_keys.extend(
                    [self.window_metadata_key, self.original_text_metadata_key, 'file_path', 'file_name', 'filename', 'extension']
                )

                all_nodes.append(node)
        return all_nodes


        
   
