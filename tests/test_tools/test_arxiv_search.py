"""Unit tests for arXiv direct search tool."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import pytest

from src.tools.arxiv_search import ArxivSearchInput, _raw_arxiv_search, arxiv_search, arxiv_search_tool


MOCK_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2404.14219v1</id>
    <published>2024-04-22T17:00:00Z</published>
    <title>Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone</title>
    <summary>We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens.</summary>
    <author><name>Marah Abdin</name></author>
    <author><name>Jyoti Aneja</name></author>
    <author><name>Hany Awadalla</name></author>
    <link href="http://arxiv.org/pdf/2404.14219v1" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>
"""


@pytest.mark.anyio
async def test_arxiv_xml_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockResponse:
        text = MOCK_ARXIV_XML
        status_code = 200

        def raise_for_status(self) -> None:
            pass

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url: str):
            return MockResponse()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    params = ArxivSearchInput(query="phi-3 technical report", max_results=1)
    evidence = await arxiv_search(params)

    assert len(evidence) == 1
    ev = evidence[0]
    assert "Phi-3 Technical Report" in ev.title
    assert "Abdin" in ev.author
    assert ev.date == "2024"
    assert "arxiv.org/abs/2404.14219" in ev.source_url
    assert ev.source_type == "academic"
    assert "3.8 billion parameter" in ev.content


@pytest.mark.anyio
async def test_arxiv_search_tool_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockResponse:
        text = MOCK_ARXIV_XML
        status_code = 200

        def raise_for_status(self) -> None:
            pass

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url: str):
            return MockResponse()

    import httpx, json
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    tool_result = await arxiv_search_tool.ainvoke({"query": "phi-3", "max_results": 1})
    parsed = json.loads(tool_result)

    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["title"].startswith("Phi-3 Technical Report")
    assert parsed[0]["year"] == "2024"
    assert "arxiv_id" in parsed[0]
