import re
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def concatenate_rows(date: str, sender: str, text: str) -> str:
    """Combine message information in a readable format ready to be used."""
    return f"{sender} on {date}: {text}\n\n"


class WhatsAppChatLoader(BaseLoader):
    """Loader that loads WhatsApp messages text file."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.file_path)
        text_content = ""

        with open(p, encoding="utf8") as f:
            lines = f.readlines()

        message_line_regex = r"""
            \[?
            (
                \d{1,4}
                [\/.]
                \d{1,2}
                [\/.]
                \d{1,4}
                ,\s
                \d{1,2}
                :\d{2}
                (?:
                    :\d{2}
                )?
                (?:[ _](?:AM|PM))?
            )
            \]?
            [\s-]*
            ([~\w\s]+)
            [:]+
            \s
            (.+)
        """
        for line in lines:
            result = re.match(message_line_regex, line.strip(), flags=re.VERBOSE)
            if result:
                date, sender, text = result.groups()
                text_content += concatenate_rows(date, sender, text)

        metadata = {"source": str(p)}

        return [Document(page_content=text_content, metadata=metadata)]
