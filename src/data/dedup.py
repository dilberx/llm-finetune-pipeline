"""
MinHash-based near-duplicate detection and removal.

Uses LSH to efficiently find and remove near-duplicate code samples.
"""

from __future__ import annotations

from datasketch import MinHash, MinHashLSH
from rich.console import Console
from rich.progress import track

console = Console()


def _tokenize(text: str, ngram_size: int = 5) -> list[str]:
    """Convert text to character n-grams (shingles)."""
    text = text.strip().lower()
    if len(text) < ngram_size:
        return [text]
    return [text[i:i + ngram_size] for i in range(len(text) - ngram_size + 1)]


def _create_minhash(tokens: list[str], num_perm: int) -> MinHash:
    """Create a MinHash signature from a list of tokens."""
    m = MinHash(num_perm=num_perm)
    for token in tokens:
        m.update(token.encode("utf-8"))
    return m


class MinHashDeduplicator:
    """Near-duplicate detector using MinHash LSH."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.7, ngram_size: int = 5):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size

    def deduplicate(self, texts: list[str]) -> list[int]:
        """Remove near-duplicates. Returns indices of unique samples."""
        if not texts:
            return []

        console.print(f"  Building MinHash signatures (n={len(texts)}, perm={self.num_perm})...")
        signatures: list[MinHash] = []
        for text in track(texts, description="  Hashing...", console=console):
            tokens = _tokenize(text, self.ngram_size)
            signatures.append(_create_minhash(tokens, self.num_perm))

        console.print(f"  Building LSH index (threshold={self.threshold})...")
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        unique_indices: list[int] = []
        duplicates_found = 0

        for idx, sig in enumerate(signatures):
            try:
                result = lsh.query(sig)
            except ValueError:
                result = []

            if not result:
                try:
                    lsh.insert(str(idx), sig)
                    unique_indices.append(idx)
                except ValueError:
                    unique_indices.append(idx)
            else:
                duplicates_found += 1

        console.print(
            f"  [green]Dedup complete:[/green] "
            f"{len(unique_indices)} unique, {duplicates_found} removed"
        )
        return unique_indices

    def estimate_similarity(self, text_a: str, text_b: str) -> float:
        """Estimate Jaccard similarity between two texts."""
        tokens_a = _tokenize(text_a, self.ngram_size)
        tokens_b = _tokenize(text_b, self.ngram_size)
        hash_a = _create_minhash(tokens_a, self.num_perm)
        hash_b = _create_minhash(tokens_b, self.num_perm)
        return hash_a.jaccard(hash_b)
