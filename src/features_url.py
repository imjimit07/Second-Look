"""
features_url.py — URL and domain risk features.
Detects: homoglyph/confusable domains, brand-in-subdomain,
suspicious TLDs, IP-literal URLs, punycode, and lexical signals.

References:
- tldextract for accurate domain parsing
- confusable_homoglyphs for Unicode confusable detection
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

import tldextract
from confusable_homoglyphs import confusables

# Lazy-load tldextract to avoid network calls on first use
_tld_extract = tldextract.TLDExtract(suffix_list_urls=())


# --- Configuration ---
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",           # Freenom free TLDs
    "xyz", "top", "info", "biz", "online",
    "site", "club", "work", "click", "link",
    "zip", "mov", "country", "stream",
}

URL_SHORTENERS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
}

SENSITIVE_SUBDOMAINS = {
    "login", "verify", "secure", "auth", "account",
    "update", "confirm", "signin", "banking", "wallet",
    "payment", "billing", "support", "help", "service",
}

SENSITIVE_PATH_KEYWORDS = {
    "login", "verify", "auth", "password", "wallet",
    "privatekey", "seed", "confirm", "update", "secure",
    "account", "billing", "payment",
}

# Common brand names targeted by typosquatting
TARGET_BRANDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "netflix",
    "chase", "wellsfargo", "hsbc", "sbi", "hdfc", "icici",
    "facebook", "instagram", "whatsapp", "linkedin", "dropbox",
]


@dataclass
class URLFeatures:
    """All extracted features for a single URL."""
    url: str
    registered_domain: str = ""
    subdomain: str = ""
    suffix: str = ""
    is_ip_literal: bool = False
    is_shortener: bool = False
    is_suspicious_tld: bool = False
    has_punycode: bool = False
    has_homoglyphs: bool = False
    brand_in_subdomain: bool = False
    brand_in_path: bool = False
    sensitive_subdomain: bool = False
    sensitive_path: bool = False
    url_length: int = 0
    domain_length: int = 0
    hyphen_count: int = 0
    digit_count: int = 0
    dot_count: int = 0
    entropy: float = 0.0
    risk_score: int = 0
    risk_details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "registered_domain": self.registered_domain,
            "subdomain": self.subdomain,
            "suffix": self.suffix,
            "is_ip_literal": self.is_ip_literal,
            "is_shortener": self.is_shortener,
            "is_suspicious_tld": self.is_suspicious_tld,
            "has_punycode": self.has_punycode,
            "has_homoglyphs": self.has_homoglyphs,
            "brand_in_subdomain": self.brand_in_subdomain,
            "brand_in_path": self.brand_in_path,
            "sensitive_subdomain": self.sensitive_subdomain,
            "sensitive_path": self.sensitive_path,
            "url_length": self.url_length,
            "domain_length": self.domain_length,
            "hyphen_count": self.hyphen_count,
            "digit_count": self.digit_count,
            "entropy": round(self.entropy, 3),
            "risk_score": self.risk_score,
            "risk_details": self.risk_details,
        }


def calculate_entropy(text: str) -> float:
    """Shannon entropy of a string. High entropy = random-looking."""
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


def check_homoglyphs(domain: str) -> bool:
    """Check if domain contains Unicode confusable characters.

    Uses is_dangerous() (not is_confusable()): every ASCII char technically
    has a confusable, so is_confusable() false-positives on plain Latin
    domains. is_dangerous with preferred_aliases=['latin'] only flags
    domains mixing in unexpected scripts (e.g. Cyrillic lookalikes).
    """
    try:
        return bool(confusables.is_dangerous(domain, preferred_aliases=["latin"]))
    except Exception:
        return False


def analyze_url(url: str) -> URLFeatures:
    """Full feature extraction for a single URL."""
    features = URLFeatures(url=url)

    # Parse
    parsed = urlparse(url)
    extracted = _tld_extract(url)

    features.registered_domain = f"{extracted.domain}.{extracted.suffix}" \
        if extracted.suffix else extracted.domain
    features.subdomain = extracted.subdomain or ""
    features.suffix = extracted.suffix or ""
    features.url_length = len(url)
    features.domain_length = len(extracted.domain)
    features.hyphen_count = extracted.domain.count("-")
    features.digit_count = sum(c.isdigit() for c in extracted.domain)
    features.dot_count = url.count(".")
    features.entropy = calculate_entropy(extracted.domain)

    # IP-literal URL
    features.is_ip_literal = bool(
        re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed.netloc)
    )
    if features.is_ip_literal:
        features.risk_score += 25
        features.risk_details.append(
            "URL uses a raw IP address instead of a domain name"
        )

    # Shortener
    full_domain = features.registered_domain.lower()
    if full_domain in URL_SHORTENERS:
        features.is_shortener = True
        features.risk_score += 15
        features.risk_details.append(
            "URL is a shortener — the real destination is hidden"
        )

    # Suspicious TLD
    if features.suffix.lower() in SUSPICIOUS_TLDS:
        features.is_suspicious_tld = True
        features.risk_score += 15
        features.risk_details.append(
            f"Domain uses a TLD frequently abused for scams (.{features.suffix})"
        )

    # Punycode
    if "xn--" in url.lower():
        features.has_punycode = True
        features.risk_score += 20
        features.risk_details.append(
            "URL contains punycode encoding — may be impersonating a brand"
        )

    # Homoglyphs
    if check_homoglyphs(features.registered_domain):
        features.has_homoglyphs = True
        features.risk_score += 25
        features.risk_details.append(
            "Domain contains Unicode confusable characters (homoglyph attack)"
        )

    # Brand in subdomain (e.g., paypal.com.secure-login.xyz)
    subdomain_lower = features.subdomain.lower()
    for brand in TARGET_BRANDS:
        if brand in subdomain_lower:
            features.brand_in_subdomain = True
            features.risk_score += 20
            features.risk_details.append(
                f"Brand name '{brand}' appears in the subdomain, not the domain"
            )
            break

    # Brand in path
    path_lower = parsed.path.lower()
    for brand in TARGET_BRANDS:
        if brand in path_lower:
            features.brand_in_path = True
            features.risk_score += 10
            features.risk_details.append(
                f"Brand name '{brand}' appears in the URL path"
            )
            break

    # Sensitive subdomain
    for sub in SENSITIVE_SUBDOMAINS:
        if sub in subdomain_lower:
            features.sensitive_subdomain = True
            features.risk_score += 10
            features.risk_details.append(
                f"Subdomain contains '{sub}' — common in credential phishing"
            )
            break

    # Sensitive path
    for kw in SENSITIVE_PATH_KEYWORDS:
        if kw in path_lower:
            features.sensitive_path = True
            features.risk_score += 10
            features.risk_details.append(
                f"URL path contains '{kw}' — targets account credentials"
            )
            break

    # Lexical signals
    if features.hyphen_count >= 3:
        features.risk_score += 10
        features.risk_details.append(
            "Domain contains many hyphens — common in auto-generated scam domains"
        )
    if features.entropy > 3.5:
        features.risk_score += 5
        features.risk_details.append(
            "Domain has high character entropy — may be auto-generated"
        )

    features.risk_score = min(features.risk_score, 100)
    return features


def analyze_urls(urls: List[str]) -> List[URLFeatures]:
    """Analyze a batch of URLs."""
    return [analyze_url(u) for u in urls]


# --- Quick test ---
if __name__ == "__main__":
    test_urls = [
        "https://www.paypal.com/signin",
        "http://paypal-secure-login.xyz/verify",
        "https://binanсe-secure.com/login",
        "http://192.168.1.1/admin",
        "https://bit.ly/3xYzAbC",
    ]
    for url in test_urls:
        feats = analyze_url(url)
        print(f"\nURL: {url}")
        print(f"  Risk: {feats.risk_score}/100")
        for detail in feats.risk_details:
            print(f"    • {detail}")
