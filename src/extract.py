"""
extract.py — Extract structured entities from raw message text.
No AI. Pure regex + heuristics. Fast, auditable, explainable.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse


# --- Regex patterns ---
URL_PATTERN = re.compile(
    r'https?://[^\s<>"\']+|'
    r'www\.[^\s<>"\']+|'
    r'\b[a-zA-Z0-9-]+\.(?:com|net|org|io|co|xyz|top|info|biz|online|site|tk|ml|ga|cf|gq)[^\s<>"\']*',
    re.IGNORECASE
)

PHONE_PATTERN = re.compile(
    r'(?:\+?\d{1,3}[\s-]?)?\(?\d{2,5}\)?[\s.-]?\d{3,5}[\s.-]?\d{3,5}'
)

AMOUNT_PATTERN = re.compile(
    r'(?:₹|Rs\.?|\$|€|£|INR|USD|EUR)\s?[\d,]+(?:\.\d{2})?'
)

EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Known brand names commonly impersonated
KNOWN_BRANDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "netflix",
    "bank", "chase", "wells fargo", "hsbc", "sbi", "hdfc", "icici",
    "irs", "social security", "medicare", "fedex", "ups", "dhl",
    "whatsapp", "instagram", "facebook", "meta", "linkedin",
]

# Urgency / pressure keywords (rule-based, not AI)
URGENCY_WORDS = [
    "urgent", "immediately", "within 24 hours", "act now", "expire",
    "suspended", "locked", "verify now", "final warning", "last chance",
    "time-sensitive", "asap", "right away", "don't delay",
    "account will be closed", "legal action", "arrest",
]

# Financial lure keywords
FINANCIAL_WORDS = [
    "refund", "prize", "winner", "lottery", "cash", "reward",
    "claim", "bonus", "deposit", "transfer", "payment", "invoice",
    "bitcoin", "crypto", "usdt", "gift card", "wire",
]

# Authority impersonation keywords
AUTHORITY_WORDS = [
    "irs", "police", "government", "social security", "medicare",
    "tax", "customs", "court", "legal notice", "fbi", "interpol",
    "bank security", "fraud department", "compliance team",
]


@dataclass
class ExtractedEntities:
    """Structured output from extraction. Every field is auditable."""
    urls: List[str] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    brands_mentioned: List[str] = field(default_factory=list)
    urgency_signals: List[str] = field(default_factory=list)
    financial_signals: List[str] = field(default_factory=list)
    authority_signals: List[str] = field(default_factory=list)
    text_length: int = 0
    word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "urls": self.urls,
            "phone_numbers": self.phone_numbers,
            "amounts": self.amounts,
            "emails": self.emails,
            "brands_mentioned": self.brands_mentioned,
            "urgency_signals": self.urgency_signals,
            "financial_signals": self.financial_signals,
            "authority_signals": self.authority_signals,
            "text_length": self.text_length,
            "word_count": self.word_count,
        }

    def risk_signal_count(self) -> int:
        """Count of independent red-flag signals detected."""
        return sum([
            len(self.urgency_signals),
            len(self.financial_signals),
            len(self.authority_signals),
            len(self.urls),
        ])


def extract_urls(text: str) -> List[str]:
    """Extract URLs, deduplicated and normalized."""
    found = URL_PATTERN.findall(text)
    seen = set()
    result = []
    for url in found:
        url = url.rstrip(".,;:!?)")
        if not url.startswith("http"):
            url = "https://" + url
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc not in seen:
            seen.add(parsed.netloc)
            result.append(url)
    return result


def extract_brands(text: str) -> List[str]:
    """Find brand names mentioned in text (case-insensitive)."""
    text_lower = text.lower()
    return [brand for brand in KNOWN_BRANDS if brand in text_lower]


def extract_keyword_hits(text: str, keywords: List[str]) -> List[str]:
    """Find which keywords from a list appear in text."""
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def extract_all(text: str) -> ExtractedEntities:
    """Main extraction function. Returns structured entities."""
    entities = ExtractedEntities(
        urls=extract_urls(text),
        phone_numbers=list(set(PHONE_PATTERN.findall(text))),
        amounts=list(set(AMOUNT_PATTERN.findall(text))),
        emails=list(set(EMAIL_PATTERN.findall(text))),
        brands_mentioned=extract_brands(text),
        urgency_signals=extract_keyword_hits(text, URGENCY_WORDS),
        financial_signals=extract_keyword_hits(text, FINANCIAL_WORDS),
        authority_signals=extract_keyword_hits(text, AUTHORITY_WORDS),
        text_length=len(text),
        word_count=len(text.split()),
    )
    return entities


# --- Quick test ---
if __name__ == "__main__":
    test_msg = (
        "URGENT: Your SBI account has been suspended. "
        "Verify immediately at http://sbi-secure-login.xyz/verify "
        "or call +91-9876543210. Failure to act within 24 hours "
        "will result in permanent closure and legal action. "
        "Claim your pending refund of Rs. 15,000."
    )
    result = extract_all(test_msg)
    import json
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nRisk signals detected: {result.risk_signal_count()}")
