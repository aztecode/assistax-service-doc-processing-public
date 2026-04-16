"""
Shared patterns for legal document parsing.
Prevents "VI" from matching inside "VIGESIMO" (Article 20 vs Article 6)
and supports written ordinals (Artículo Vigésimo, Décimo Tercero, etc.).
Ported from assistax-back/src/services/legalOrdinalPatterns.ts
"""

# Roman numerals with negative lookahead: prevents substrings like "VI" in "VIGESIMO"
ROMAN_WITH_LOOKAHEAD = (
    r"(?:[IVX]+|M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))(?!\w)"
)

# Ordinal words for article numbers (1-9, 10-19, 20-29).
# Order: compound ordinals first (longest), then simple.
# Accent variants: [eé], [ií], [oó], [uú] for documents that omit accents.
ARTICLE_ORDINAL_WORDS = (
    # 21-29: vigésimo primero ... vigésimo noveno
    r"vig[eé]simo\s+noveno",
    r"vig[eé]simo\s+octavo",
    r"vig[eé]simo\s+s[eé]ptimo",
    r"vig[eé]simo\s+sexto",
    r"vig[eé]simo\s+quinto",
    r"vig[eé]simo\s+cuarto",
    r"vig[eé]simo\s+tercero",
    r"vig[eé]simo\s+segundo",
    r"vig[eé]simo\s+primero",
    r"vig[eé]simo",
    # 13-19: décimo tercero ... décimo noveno
    r"d[eé]cimo\s+noveno",
    r"d[eé]cimo\s+octavo",
    r"d[eé]cimo\s+s[eé]ptimo",
    r"d[eé]cimo\s+sexto",
    r"d[eé]cimo\s+quinto",
    r"d[eé]cimo\s+cuarto",
    r"d[eé]cimo\s+tercero",
    # 11-12
    r"und[eé]cimo",
    r"duod[eé]cimo",
    # 10
    r"d[eé]cimo",
    # 1-9
    r"primero",
    r"segundo",
    r"tercero",
    r"cuarto",
    r"quinto",
    r"sexto",
    r"s[eé]ptimo",
    r"octavo",
    r"noveno",
    r"[uú]nico",
)

# Joined for use in regex patterns
ARTICLE_ORDINAL_WORDS_PATTERN = "|".join(ARTICLE_ORDINAL_WORDS)
