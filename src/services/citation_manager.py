"""Citation management service for tracking, resolving, and formatting citations."""

from __future__ import annotations

import re
from typing import Any

from src.models.state import Citation, Evidence


class CitationManager:
    """Manages citation registration, resolution, deduplication, and bibliography generation.

    Maintains an internal registry mapping source URLs to Citation objects,
    enabling consistent citation numbering across the entire research report.
    """

    def __init__(self) -> None:
        """Initialize the citation manager with an empty registry."""
        self._registry: dict[str, Citation] = {}
        self._next_number: int = 1

    def register_evidence(self, evidence: list[Evidence]) -> None:
        """Register all evidence items, creating Citation entries for each unique source.

        Args:
            evidence: List of Evidence objects to register as citable sources.
        """
        for item in evidence:
            url = item.source_url.strip() if item.source_url else ""
            if not url:
                continue
            if url not in self._registry:
                # Determine best author string
                author = item.author if hasattr(item, "author") and item.author and item.author != "Unknown" else ""
                if not author:
                    authors = item.metadata.get("authors", []) if hasattr(item, "metadata") else []
                    if authors:
                        author = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
                    elif getattr(item, "source_type", "") == "web":
                        author = "Web Source"
                    else:
                        author = "Unknown"

                date = item.date if hasattr(item, "date") and item.date else str(item.metadata.get("year", "")) if hasattr(item, "metadata") else ""

                self._registry[url] = Citation(
                    number=self._next_number,
                    source_url=url,
                    title=item.title.strip() if item.title else "Untitled",
                    author=author,
                    date=date,
                    snippet=item.content[:300] if item.content else "",
                    source_type=getattr(item, "source_type", "web"),
                )
                self._next_number += 1

    def resolve_citations(
        self, content: str, evidence: list[Evidence]
    ) -> tuple[str, list[Citation]]:
        """Resolve inline citation markers (e.g., [1], [2]) to actual citations.

        Correctly maps local section evidence index [1]..[K] to global citation
        numbers and registers sources into the shared bibliography.

        Args:
            content: Text content containing inline citation markers like [1], [2].
            evidence: List of Evidence objects provided to the section synthesizer.

        Returns:
            A tuple of (updated_content, referenced_citations).
        """
        citation_pattern = re.compile(r"\[(\d+)\]")
        found_numbers: list[int] = []
        for match in citation_pattern.finditer(content):
            num = int(match.group(1))
            if num not in found_numbers:
                found_numbers.append(num)

        number_mapping: dict[int, int] = {}
        referenced_citations: list[Citation] = []

        for old_num in found_numbers:
            target_evidence: Evidence | None = None
            if 1 <= old_num <= len(evidence):
                target_evidence = evidence[old_num - 1]

            if target_evidence and target_evidence.source_url:
                url = target_evidence.source_url.strip()
                if url not in self._registry:
                    self.register_evidence([target_evidence])
                citation = self._registry.get(url)
                if citation:
                    number_mapping[old_num] = citation.number
                    if citation not in referenced_citations:
                        referenced_citations.append(citation)
            else:
                # If old_num matches an existing global citation, preserve it
                matching_global = next((c for c in self._registry.values() if c.number == old_num), None)
                if matching_global:
                    number_mapping[old_num] = matching_global.number
                    if matching_global not in referenced_citations:
                        referenced_citations.append(matching_global)

        def replace_citation(match: re.Match[str]) -> str:
            old_n = int(match.group(1))
            if old_n in number_mapping:
                return f"[{number_mapping[old_n]}]"
            return match.group(0)

        updated_content = citation_pattern.sub(replace_citation, content)
        referenced_citations.sort(key=lambda c: c.number)

        return updated_content, referenced_citations

    def build_bibliography(self, citations: list[Citation]) -> str:
        """Format a list of citations as a numbered markdown bibliography.

        Args:
            citations: List of Citation objects to format.

        Returns:
            A formatted markdown string containing the bibliography.
        """
        if not citations:
            return ""

        lines: list[str] = ["## Bibliography", ""]
        deduplicated = sorted(self.deduplicate(citations), key=lambda c: c.number)

        for citation in deduplicated:
            author_part = f"{citation.author}. " if citation.author and citation.author != "Unknown" else ""
            date_part = f" ({citation.date})." if citation.date else "."
            title_part = f' "{citation.title}".' if citation.title else ' "Untitled".'

            arxiv_part = ""
            if "arxiv.org/abs/" in citation.source_url:
                arxiv_id = citation.source_url.split("/abs/")[-1]
                arxiv_part = f" arXiv:{arxiv_id}."

            url_part = f" Available at: {citation.source_url}" if citation.source_url else ""

            lines.append(
                f"[{citation.number}] {author_part}{title_part}{date_part}{arxiv_part}{url_part}"
            )

        lines.append("")
        return "\n".join(lines)

    def deduplicate(self, citations: list[Citation]) -> list[Citation]:
        """Remove duplicate citations based on source_url.

        Keeps the first occurrence (lowest citation number) for each unique URL.

        Args:
            citations: List of Citation objects, potentially containing duplicates.

        Returns:
            Deduplicated list of Citation objects, preserving original order.
        """
        seen_urls: set[str] = set()
        unique: list[Citation] = []

        for citation in citations:
            if citation.source_url not in seen_urls:
                seen_urls.add(citation.source_url)
                unique.append(citation)

        return unique
