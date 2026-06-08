"""CBFC Gazette pipeline.

Searches the Internet Archive `gazetteofindia` collection for fulltext mentions
of the Central Board of Film Certification (and its pre-1983 name, Central
Board of Film Censors), downloads every matching gazette's OCR text, asks
Gemini to extract any film-certification entries with alterations into a
structured JSON, then writes a single CSV.

Resumable at every stage: cached index, cached downloads, cached LLM JSON.
Re-running picks up where it left off.

Run: `uv run python cbfc.py` (needs GEMINI_API_KEY in env).
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from google import genai
from google.genai import types
from tqdm import tqdm

FTS_URL = "https://be-api.us.archive.org/ia-pub-fts-api"
DOWNLOAD_URL = "https://archive.org/download"
DETAILS_URL = "https://archive.org/details"
COLLECTION = "gazetteofindia"
# CBFC was renamed from "Central Board of Film Censors" → "Central Board of
# Film Certification" in 1983. Search both phrases so we don't silently drop
# the entire pre-1983 era (Lawrence-of-Arabia etc.).
QUERIES = [
    '"Central Board of Film Certification"',
    '"Central Board of Film Censors"',
]
PAGE_SIZE = 100
DL_CONCURRENCY = 8
LLM_CONCURRENCY = 6
GEMINI_MODEL = "gemini-2.5-flash-lite"
HTTP_TIMEOUT = httpx.Timeout(60.0, connect=15.0)

# Strong CBFC markers used to skip admin/banking gazettes that mention the
# Board only in passing. Anything without at least one of these is never sent
# to the LLM.
MARKER_RE = re.compile(
    r"(?i)(?:Alterations?\s+under\s+Rule"
    r"|Length\s+of\s+(?:deletions?|voluntary|approved)"
    r"|Actual\s+length\s+of\s+the\s+\w{3,5})"
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"
LLM_OUT = DATA / "llm"
INDEX_PATH = DATA / "index.jsonl"
CSV_PATH = DATA / "movies.csv"
COMPLETED_PATH = DATA / ".completed.txt"


@dataclass
class Item:
    identifier: str
    title: str
    date: str
    file_basename: str

    @property
    def text_url(self) -> str:
        return f"{DOWNLOAD_URL}/{self.identifier}/{self.file_basename}_djvu.txt"

    @property
    def details_url(self) -> str:
        return f"{DETAILS_URL}/{self.identifier}"

    @property
    def cache_path(self) -> Path:
        return RAW / f"{self.identifier}.txt"

    @property
    def result_path(self) -> Path:
        return LLM_OUT / f"{self.identifier}.json"


def _first(v: object) -> str:
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v) if v is not None else ""


# --- Stage 1: search ----------------------------------------------------------

async def search_items(client: httpx.AsyncClient) -> list[Item]:
    seen: dict[str, Item] = {}
    for query in QUERIES:
        offset = 0
        total: int | None = None
        pbar: tqdm | None = None
        while True:
            r = await client.get(
                FTS_URL,
                params={"q": query, "collection": COLLECTION, "size": PAGE_SIZE, "from": offset},
            )
            r.raise_for_status()
            body = r.json()
            if total is None:
                total = int(body["hits"]["total"])
                pbar = tqdm(total=total, desc=f"index[{query[:24]}…]", unit="hit")
            hits = body["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                f = h.get("fields") or {}
                ident = _first(f.get("identifier"))
                if not ident or ident in seen:
                    continue
                seen[ident] = Item(
                    identifier=ident,
                    title=_first(f.get("meta_title")),
                    date=_first(f.get("meta_date")),
                    file_basename=f.get("file_basename") or ident,
                )
            offset += len(hits)
            if pbar:
                pbar.update(len(hits))
            if offset >= total:
                break
        if pbar:
            pbar.close()
    return list(seen.values())


# --- Stage 2: download --------------------------------------------------------

async def _download_one(client: httpx.AsyncClient, item: Item, sem: asyncio.Semaphore) -> None:
    if item.cache_path.exists():
        return
    async with sem:
        for attempt in range(3):
            try:
                r = await client.get(item.text_url)
                if r.status_code == 200:
                    item.cache_path.write_bytes(r.content)
                    return
                if r.status_code == 404:
                    item.cache_path.write_bytes(b"")
                    return
            except httpx.HTTPError:
                await asyncio.sleep(2**attempt)
        item.cache_path.write_bytes(b"")


async def download_all(items: list[Item]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(DL_CONCURRENCY)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        tasks = [_download_one(client, it, sem) for it in items]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="download", unit="doc"):
            await fut


# --- Stage 3: LLM extract -----------------------------------------------------

SYSTEM_PROMPT = """You extract Central Board of Film Certification (CBFC) records from OCR'd Indian government gazettes (Gazette of India, Part III, Section 4).

The OCR is noisy. Em-dashes (—) frequently render as one or more hyphens. Decimal points may have whitespace ("2664 . 26") or appear as hyphens ("2663 -58" means 2663.58). Letters get mangled ("filn"→"film", "Evglish"→"English", "addrossed"→"addressed", "datod"→"dated", "Boauty"→"Beauty"). Table layouts are sometimes scrambled across non-adjacent lines.

Each CBFC entry contains:
- a film title, usually in quotes near "Film —", "Film :" or "Film ;"
- one or more parenthesised tokens like (English), (Hindi), (Colour), (Cinemascope), (70 mm)
- a producer / distributor (production company name with address)
- a certificate ref like "U-Cert. No. 43904 dt. 21-1-1965" or "A-Cert. No. 1563, dated 14-1-1965". The prefix letter (U/A/B/UA/S) is the certificate category; B is restricted, A is adult, U is universal, UA is parental guidance, S is specialised.
- "Alterations under Rule 34" introducing one or more alteration sections:
    * "Delete" / "Deleted" — content cut entirely
    * "Reduce" / "Reduced" — content shortened (often gives both deleted length and retained length)
    * "Inserted" — content added
    * Replacements (sometimes phrased "Replaced ... with ...")
  Each section often lists per-reel descriptions: "Reel VII (4A) — ..." with a length in metres.
- A final "Actual length of the film after the aforesaid alterations will be — N.NN m." line.

Extract EVERY entry that has at least one alteration (deletion, reduction, insertion, or replacement). Skip plain certifications with no cuts. Skip non-film admin/banking/recruitment content even if it mentions the Board.

For each alteration, give:
  - kind: one of "deletion", "reduction", "insertion", "replacement"
  - reel: the reel reference if stated (e.g. "Reel VII", "XB", "XII A"); empty string if absent
  - description: a concise, plain-English summary of WHAT was cut/changed. Do NOT verbatim-copy OCR garbage; clean it. Example: instead of "tho dialogue of Din Dayal Hamarc Nazrome police our Kuttc dono ek hain" → "the dialogue of Din Dayal comparing police to dogs".
  - length_m: deleted/inserted length in metres if given; omit if illegible
  - length_retained_m: only for reductions, the length kept after the cut; omit if not given

Normalise all lengths to metres. If a length is given in feet ("ft.") or feet+frames, convert to metres (1 ft ≈ 0.3048 m) and put the metres value.

Clean up obvious OCR typos in the film_title (e.g. "Boauty" → "Beauty", "filn" → "Film", "I d" → "I'd") but keep the title in its original language/spelling otherwise. Do not invent details that aren't in the text.

Return a JSON array of entries; return [] if the document has no CBFC alteration entries.
"""

RESULT_SCHEMA = types.Schema(
    type="ARRAY",
    items=types.Schema(
        type="OBJECT",
        required=["film_title", "alterations"],
        properties={
            "film_title": types.Schema(type="STRING"),
            "languages": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
            "colour": types.Schema(type="STRING"),
            "format": types.Schema(type="STRING"),
            "producer": types.Schema(type="STRING"),
            "cert_no": types.Schema(type="STRING"),
            "cert_date": types.Schema(type="STRING"),
            "endorsement": types.Schema(
                type="STRING",
                description="One of U, A, B, UA, S (the certificate category letter).",
            ),
            "rule_applied": types.Schema(type="STRING"),
            "original_length_m": types.Schema(type="NUMBER"),
            "length_deleted_m": types.Schema(type="NUMBER"),
            "final_length_m": types.Schema(type="NUMBER"),
            "alterations": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    required=["kind", "description"],
                    properties={
                        "kind": types.Schema(
                            type="STRING",
                            enum=["deletion", "reduction", "insertion", "replacement"],
                        ),
                        "reel": types.Schema(type="STRING"),
                        "description": types.Schema(type="STRING"),
                        "length_m": types.Schema(type="NUMBER"),
                        "length_retained_m": types.Schema(type="NUMBER"),
                    },
                ),
            ),
        },
    ),
)


def _load_completed() -> set[str]:
    if not COMPLETED_PATH.exists():
        return set()
    return {ln.strip() for ln in COMPLETED_PATH.read_text().splitlines() if ln.strip()}


def _append_completed(ident: str) -> None:
    COMPLETED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPLETED_PATH.open("a") as f:
        f.write(ident + "\n")
        f.flush()


def _extract_one(client: genai.Client, item: Item, text: str) -> list[dict] | None:
    """Send one gazette's OCR text to Gemini; return list of entries (possibly empty)."""
    # Flash-Lite has a 1M-token context, but capping bytes keeps cost predictable
    # and avoids occasional timeouts on outlier huge docs.
    if len(text) > 600_000:
        text = text[:600_000]
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESULT_SCHEMA,
                system_instruction=types.Part.from_text(text=SYSTEM_PROMPT),
            ),
        )
    except Exception as e:
        print(f"  ! {item.identifier}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    raw = getattr(resp, "text", "") or ""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ! {item.identifier}: JSON decode error: {e}", file=sys.stderr)
        return None
    return data if isinstance(data, list) else []


def extract_all(items: list[Item]) -> None:
    LLM_OUT.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("ERROR: GEMINI_API_KEY env var not set")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    completed = _load_completed()

    candidates: list[tuple[Item, str]] = []
    skipped_no_marker = 0
    for it in items:
        if it.identifier in completed:
            continue
        if not it.cache_path.exists() or it.cache_path.stat().st_size == 0:
            continue
        text = it.cache_path.read_text(encoding="utf-8", errors="replace")
        if MARKER_RE.search(text):
            candidates.append((it, text))
        else:
            skipped_no_marker += 1

    print(
        f"extract: {len(candidates)} candidate docs "
        f"({len(completed)} already done, {skipped_no_marker} skipped (no CBFC markers))"
    )
    if not candidates:
        return

    def task(item_text: tuple[Item, str]) -> tuple[Item, list[dict] | None]:
        it, text = item_text
        return it, _extract_one(client, it, text)

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futures = {pool.submit(task, ct): ct[0] for ct in candidates}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM extract", unit="doc"):
            it = futures[fut]
            try:
                _, entries = fut.result()
            except Exception as e:
                print(f"  ! {it.identifier}: pool error: {e}", file=sys.stderr)
                continue
            if entries is None:
                continue  # transient error — try again next run
            it.result_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
            _append_completed(it.identifier)


# --- Stage 4: CSV -------------------------------------------------------------

CSV_COLUMNS = [
    "film_title", "languages", "colour", "format", "producer",
    "cert_no", "cert_date", "endorsement", "rule_applied",
    "original_length_m", "length_deleted_m", "final_length_m",
    "deletions", "reductions", "insertions", "replacements",
    "gazette_identifier", "gazette_date", "source_url",
]


def _fmt_alt(a: dict) -> str:
    parts: list[str] = []
    reel = (a.get("reel") or "").strip()
    if reel:
        parts.append(f"[{reel}]")
    desc = (a.get("description") or "").strip()
    if desc:
        parts.append(desc)
    length = a.get("length_m")
    if isinstance(length, (int, float)):
        parts.append(f"({length:g} m)")
    retained = a.get("length_retained_m")
    if isinstance(retained, (int, float)):
        parts.append(f"(retained: {retained:g} m)")
    return " ".join(parts)


def _fmt_num(v: object) -> str:
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return ""


def _norm_title(t: str) -> str:
    """Normalise a film title for dedup: strip quotes, lowercase, collapse whitespace,
    drop a leading 'Trailer of ' so trailer/film pairs can be compared."""
    s = re.sub(r"\s+", " ", t).strip().strip("\"'“”‘’").lower()
    s = re.sub(r"^trailer\s+(?:no\.?\s*\d+\s+)?(?:of\s+)?", "", s)
    return s.strip(" \"'“”‘’")


def _norm_cuts(s: str) -> str:
    """Fingerprint of a cuts text for fuzzy dedup. Extracts each ≥4-char
    alphabetic token, truncates to its first 4 chars, takes the first 20 — so
    reel refs ([I] / [1]), numbers, lengths, and OCR typos at the tail of a
    word (dialogue → dialoguc both become 'dial') don't break the comparison."""
    stems = [w[:4] for w in re.findall(r"[a-zA-Z]{4,}", s.lower())]
    return " ".join(stems[:20])


def _is_trailer(title: str) -> bool:
    return title.lower().lstrip(" \"'“”‘’").startswith("trailer")


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop redundant rows from gazette republishes, endorsement repeats, and
    LLM contamination across adjacent OCR entries.

    Pass A — trailer kill: drop "Trailer of X" rows whose cuts fingerprint
    matches a non-trailer X under the same normalised title. These are
    almost always LLM contamination from adjacent OCR table cells.

    Pass B — cross-gazette collapse: collapse by (norm_title, cuts_sig). Prefer
    the row with (i) an all-digit primary cert_no over a B/...-prefixed
    endorsement, (ii) a populated gazette_date, (iii) the earliest date.
    """
    def cuts_sig(r: dict[str, str]) -> str:
        body = r["deletions"] or (r["reductions"] + r["insertions"])
        return _norm_cuts(body)

    # Pass A
    non_trailer_cuts: dict[str, set[str]] = {}
    for r in rows:
        if not _is_trailer(r["film_title"]):
            non_trailer_cuts.setdefault(_norm_title(r["film_title"]), set()).add(cuts_sig(r))
    after_a = [
        r for r in rows
        if not (_is_trailer(r["film_title"]) and cuts_sig(r) in non_trailer_cuts.get(_norm_title(r["film_title"]), set()))
    ]

    # Pass B — for each key, pick the highest-scored candidate.
    def score(r: dict[str, str]) -> tuple:
        cert = r["cert_no"].strip()
        all_digits = cert.isdigit() and len(cert) >= 3
        has_date = bool(r["gazette_date"])
        # Higher tuple wins. Use negative date so "earlier is better" stays
        # consistent with max().
        return (all_digits, has_date, -ord(r["gazette_date"][:1] or "z"))

    best: dict[tuple[str, str], dict[str, str]] = {}
    for r in after_a:
        key = (_norm_title(r["film_title"]), cuts_sig(r))
        prev = best.get(key)
        if prev is None or score(r) > score(prev):
            best[key] = r
        elif score(r) == score(prev):
            # Stable tie-break: earlier gazette_date wins.
            if (r["gazette_date"] or "9999") < (prev["gazette_date"] or "9999"):
                best[key] = r
    return list(best.values())


def export_csv(items: list[Item]) -> int:
    by_id = {it.identifier: it for it in items}
    rows: list[dict[str, str]] = []
    for rp in sorted(LLM_OUT.glob("*.json")):
        ident = rp.stem
        try:
            entries = json.loads(rp.read_text())
        except json.JSONDecodeError:
            continue
        item = by_id.get(ident)
        for entry in entries:
            title = (entry.get("film_title") or "").strip()
            if not title:
                continue
            buckets: dict[str, list[str]] = {
                "deletion": [], "reduction": [], "insertion": [], "replacement": [],
            }
            for a in entry.get("alterations") or []:
                kind = a.get("kind") or "deletion"
                buckets.setdefault(kind, []).append(_fmt_alt(a))
            if not any(buckets.values()):
                continue
            rows.append({
                "film_title": title,
                "languages": ", ".join(entry.get("languages") or []),
                "colour": entry.get("colour", "") or "",
                "format": entry.get("format", "") or "",
                "producer": entry.get("producer", "") or "",
                "cert_no": entry.get("cert_no", "") or "",
                "cert_date": entry.get("cert_date", "") or "",
                "endorsement": entry.get("endorsement", "") or "",
                "rule_applied": entry.get("rule_applied", "") or "",
                "original_length_m": _fmt_num(entry.get("original_length_m")),
                "length_deleted_m": _fmt_num(entry.get("length_deleted_m")),
                "final_length_m": _fmt_num(entry.get("final_length_m")),
                "deletions": " || ".join(buckets["deletion"]),
                "reductions": " || ".join(buckets["reduction"]),
                "insertions": " || ".join(buckets["insertion"]),
                "replacements": " || ".join(buckets["replacement"]),
                "gazette_identifier": ident,
                "gazette_date": (item.date if item else ""),
                "source_url": (item.details_url if item else f"{DETAILS_URL}/{ident}"),
            })
    before = len(rows)
    rows = _dedupe(rows)
    print(f"dedupe: {before} -> {len(rows)} rows ({before - len(rows)} dropped)")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


# --- Orchestration ------------------------------------------------------------

async def amain() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists() and INDEX_PATH.stat().st_size > 0:
        items = [Item(**json.loads(l)) for l in INDEX_PATH.read_text().splitlines() if l.strip()]
        print(f"loaded {len(items)} items from cached index")
    else:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            items = await search_items(client)
        INDEX_PATH.write_text(
            "\n".join(json.dumps(asdict(it)) for it in items) + "\n", encoding="utf-8"
        )
        print(f"indexed {len(items)} items → {INDEX_PATH}")

    await download_all(items)
    extract_all(items)
    n = export_csv(items)
    print(f"wrote {n} rows → {CSV_PATH}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
