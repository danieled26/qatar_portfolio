import re
import time
from datetime import datetime
from urllib.parse import urlencode
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd
import requests
from bs4 import BeautifulSoup

# =========================
# CONFIG
# =========================
BASE_URL = "https://www.airlinequality.com/airline-reviews/qatar-airways"
OUTPUT_CSV = "qatar_airline_reviews.csv"

# Review date used for stopping pagination (safe lower bound)
REVIEW_STOP_DATE = datetime(2016, 5, 1)

# Final filter is based on trip month
TRIP_START_MONTH = datetime(2016, 5, 1)
TRIP_END_MONTH = datetime(2026, 5, 1)

# Optional speed-up. If Skytrax honors this param, fewer pages will be scraped.
PAGESIZE = 100

REQUEST_DELAY_SECONDS = 1.0
TIMEOUT = 30
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# =========================
# SCHEMA
# =========================
COLUMNS = [
    "airline",
    "review_date",         # cleaned display format, e.g. 11 January 2026
    "review_date_iso",     # helper field, e.g. 2026-01-11
    "trip_month",          # e.g. January 2026
    "reviewer_name",
    "reviewer_country",
    "review_title",
    "overall_rating_10",
    "trip_verified",
    "review_text",
    "aircraft",
    "type_of_traveller",
    "seat_type",
    "route",
    "date_flown",
    "seat_comfort",
    "cabin_staff_service",
    "food_beverages",
    "inflight_entertainment",
    "ground_service",
    "wifi_connectivity",
    "value_for_money",
    "recommended",
    "source_url",
    "page_number",
    "scrape_order",
]

DEFAULT_VALUE = "Not Rated"

# Map the visible table labels to your output schema
LABEL_MAP = {
    "aircraft": "aircraft",
    "type of traveller": "type_of_traveller",
    "seat type": "seat_type",
    "route": "route",
    "date flown": "date_flown",
    "seat comfort": "seat_comfort",
    "cabin staff service": "cabin_staff_service",
    "food & beverages": "food_beverages",
    "inflight entertainment": "inflight_entertainment",
    "ground service": "ground_service",
    "wifi & connectivity": "wifi_connectivity",
    "value for money": "value_for_money",
    "recommended": "recommended",
}


# =========================
# HELPERS
# =========================
def clean_text(text):
    if text is None:
        return DEFAULT_VALUE
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else DEFAULT_VALUE


def remove_ordinal_suffix(date_text):
    """
    Converts:
    11th January 2026 -> 11 January 2026
    1st January 2026 -> 1 January 2026
    """
    if not date_text or date_text == DEFAULT_VALUE:
        return DEFAULT_VALUE
    return re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", date_text)


def parse_review_date_display(display_date_text):
    """
    Output keeps full month name and day:
    11 January 2026
    """
    cleaned = remove_ordinal_suffix(clean_text(display_date_text))
    if cleaned == DEFAULT_VALUE:
        return DEFAULT_VALUE
    try:
        dt = datetime.strptime(cleaned, "%d %B %Y")
        return dt.strftime("%d %B %Y").lstrip("0")
    except Exception:
        return cleaned


def parse_iso_date(iso_text):
    if not iso_text or iso_text == DEFAULT_VALUE:
        return None
    try:
        return datetime.strptime(iso_text, "%Y-%m-%d")
    except Exception:
        return None


def parse_month_year(month_year_text):
    """
    Converts:
    January 2026 -> datetime(2026,1,1)
    """
    if not month_year_text or month_year_text == DEFAULT_VALUE:
        return None
    try:
        return datetime.strptime(month_year_text.strip(), "%B %Y")
    except Exception:
        return None


def format_month_year(dt):
    if dt is None:
        return DEFAULT_VALUE
    return dt.strftime("%B %Y")


def derive_trip_month(date_flown_text, review_date_dt):
    """
    Rule:
    - use Date Flown if present
    - else use review date month
    """
    date_flown_dt = parse_month_year(date_flown_text)
    if date_flown_dt is not None:
        return date_flown_dt
    if review_date_dt is not None:
        return datetime(review_date_dt.year, review_date_dt.month, 1)
    return None


def count_filled_stars(td):
    if td is None:
        return DEFAULT_VALUE
    filled = td.select("span.star.fill")
    return str(len(filled)) if filled else DEFAULT_VALUE


def extract_country(sub_header_text, reviewer_name, display_date):
    """
    Example sub-header text:
    '151 reviews Michael Schade (Thailand) 8th January 2026'
    or
    'Iman Yusuf (Austria) 11th January 2026'
    """
    if not sub_header_text or reviewer_name == DEFAULT_VALUE or display_date == DEFAULT_VALUE:
        # fallback: just try first (...) group
        m = re.search(r"\((.*?)\)", sub_header_text or "")
        return clean_text(m.group(1)) if m else DEFAULT_VALUE

    # Use raw display date with ordinal, because that is how it appears in the subheader text
    pattern = re.escape(reviewer_name) + r"\s*\((.*?)\)\s*" + re.escape(display_date)
    m = re.search(pattern, sub_header_text)
    if m:
        return clean_text(m.group(1))

    # fallback
    m = re.search(r"\((.*?)\)", sub_header_text)
    return clean_text(m.group(1)) if m else DEFAULT_VALUE


# def extract_trip_verified_and_text(review_body_text):
#     """
#     Examples:
#     '✅ Trip Verified | Some text...'
#     'Not Verified | Some text...'
#     """
#     txt = clean_text(review_body_text)
#     if txt == DEFAULT_VALUE:
#         return DEFAULT_VALUE, DEFAULT_VALUE

#     trip_verified = DEFAULT_VALUE
#     cleaned_review_text = txt

#     if "Trip Verified" in txt:
#         trip_verified = "Yes"
#         cleaned_review_text = txt.split("|", 1)[1].strip() if "|" in txt else txt
#     elif "Not Verified" in txt:
#         trip_verified = "No"
#         cleaned_review_text = txt.split("|", 1)[1].strip() if "|" in txt else txt
#     else:
#         # if the marker is absent, leave as Not Rated
#         trip_verified = DEFAULT_VALUE

#     cleaned_review_text = clean_text(cleaned_review_text)
#     return trip_verified, cleaned_review_text

def extract_trip_verified_and_text(review_body_text):
    txt = clean_text(review_body_text)
    if txt == DEFAULT_VALUE:
        return DEFAULT_VALUE, DEFAULT_VALUE

    trip_verified = DEFAULT_VALUE
    cleaned_review_text = txt

    # ✅ Handle BOTH formats
    if "Trip Verified" in txt or "Verified Review" in txt:
        trip_verified = "Yes"
        cleaned_review_text = txt.split("|", 1)[1].strip() if "|" in txt else txt

    elif "Not Verified" in txt:
        trip_verified = "No"
        cleaned_review_text = txt.split("|", 1)[1].strip() if "|" in txt else txt

    else:
        trip_verified = DEFAULT_VALUE

    return trip_verified, clean_text(cleaned_review_text)

def build_page_url(page_num):
    return f"{BASE_URL}/page/{page_num}/"


def fetch_page(session, page_num):
    page_url = build_page_url(page_num)
    params = {"pagesize": PAGESIZE}
    final_url = page_url + "?" + urlencode(params)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(page_url, params=params, timeout=TIMEOUT, verify = False)
            resp.raise_for_status()
            return resp.text, resp.url
        except Exception as e:
            print(f"[WARN] Page {page_num} attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 * attempt)

    return None, final_url


def parse_review_card(card, page_number, source_url, scrape_order):
    row = {col: DEFAULT_VALUE for col in COLUMNS}
    row["airline"] = "Qatar Airways"
    row["page_number"] = page_number
    row["source_url"] = source_url
    row["scrape_order"] = scrape_order

    # Review date (ISO from meta or time datetime)
    date_meta = card.select_one('meta[itemprop="datePublished"]')
    time_tag = card.select_one('time[itemprop="datePublished"]')

    review_date_iso_raw = date_meta.get("content", "").strip() if date_meta else ""
    if not review_date_iso_raw and time_tag:
        review_date_iso_raw = time_tag.get("datetime", "").strip()

    review_date_dt = parse_iso_date(review_date_iso_raw)
    row["review_date_iso"] = review_date_iso_raw if review_date_iso_raw else DEFAULT_VALUE

    # Display date
    display_date_raw = time_tag.get_text(" ", strip=True) if time_tag else DEFAULT_VALUE
    row["review_date"] = parse_review_date_display(display_date_raw)

    # Rating /10
    rating_tag = card.select_one('div[itemprop="reviewRating"] span[itemprop="ratingValue"]')
    row["overall_rating_10"] = clean_text(rating_tag.get_text(strip=True)) if rating_tag else DEFAULT_VALUE

    # Title
    title_tag = card.select_one("h2.text_header")
    row["review_title"] = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else DEFAULT_VALUE

    # Reviewer name
    name_tag = card.select_one('span[itemprop="name"]')
    reviewer_name = clean_text(name_tag.get_text(" ", strip=True)) if name_tag else DEFAULT_VALUE
    row["reviewer_name"] = reviewer_name

    # Reviewer country
    sub_header = card.select_one("h3.text_sub_header.userStatusWrapper")
    sub_header_text = sub_header.get_text(" ", strip=True) if sub_header else ""
    row["reviewer_country"] = extract_country(sub_header_text, reviewer_name, display_date_raw)

    # Review body
    body_tag = card.select_one('div.text_content[itemprop="reviewBody"]')
    review_body_text = body_tag.get_text(" ", strip=True) if body_tag else DEFAULT_VALUE
    trip_verified, review_text = extract_trip_verified_and_text(review_body_text)
    row["trip_verified"] = trip_verified
    row["review_text"] = review_text

    # Ratings / attributes table
    table = card.select_one("table.review-ratings")
    if table:
        for tr in table.select("tr"):
            header_td = tr.select_one("td.review-rating-header")
            if not header_td:
                continue

            label = clean_text(header_td.get_text(" ", strip=True)).lower()

            # Normalize HTML entities / variants
            label = (
                label.replace("&amp;", "&")
                .replace("wifi and connectivity", "wifi & connectivity")
                .replace("food and beverages", "food & beverages")
            )

            mapped_col = LABEL_MAP.get(label)
            if not mapped_col:
                continue

            # Star row
            star_td = tr.select_one("td.review-rating-stars")
            if star_td:
                row[mapped_col] = count_filled_stars(star_td)
                continue

            # Text / yes-no row
            value_td = tr.select_one("td.review-value")
            if value_td:
                value_text = clean_text(value_td.get_text(" ", strip=True))

                if mapped_col == "recommended":
                    if value_text.lower() == "yes":
                        row[mapped_col] = "Yes"
                    elif value_text.lower() == "no":
                        row[mapped_col] = "No"
                    else:
                        row[mapped_col] = DEFAULT_VALUE
                else:
                    row[mapped_col] = value_text

    # Derive trip month
    trip_month_dt = derive_trip_month(row["date_flown"], review_date_dt)
    row["trip_month"] = format_month_year(trip_month_dt)

    return row, review_date_dt, trip_month_dt


def make_dedup_key(df):
    return (
        df["review_title"].fillna("").str.strip().str.lower() + "||" +
        df["reviewer_name"].fillna("").str.strip().str.lower() + "||" +
        df["review_date"].fillna("").str.strip().str.lower()
    )


# =========================
# MAIN
# =========================
def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    page_number = 1
    scrape_order = 1

    while True:
        print(f"Scraping page {page_number}...")
        html, resolved_url = fetch_page(session, page_number)
        soup = BeautifulSoup(html, "html.parser")

        # IMPORTANT: only real review articles
        review_cards = soup.select('article[itemprop="review"]')

        if not review_cards:
            print("No review cards found. Stopping.")
            break

        page_review_dates = []

        for card in review_cards:
            row, review_date_dt, trip_month_dt = parse_review_card(
                card=card,
                page_number=page_number,
                source_url=resolved_url,
                scrape_order=scrape_order,
            )
            all_rows.append(row)
            scrape_order += 1

            if review_date_dt is not None:
                page_review_dates.append(review_date_dt)

        if not page_review_dates:
            print("No parseable review dates found on page. Stopping.")
            break

        oldest_review_date_on_page = min(page_review_dates)

        # Since pages are newest -> oldest, once the page dips below the review stop date,
        # parse this page but stop fetching the next page.
        if oldest_review_date_on_page < REVIEW_STOP_DATE:
            print("Reached review-date lower bound. Stopping pagination.")
            break

        page_number += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    # Build dataframe
    df = pd.DataFrame(all_rows, columns=COLUMNS)

    # Parse trip month to datetime helper for filtering
    def trip_month_to_dt(val):
        try:
            return datetime.strptime(val, "%B %Y")
        except Exception:
            return None

    df["_trip_month_dt"] = df["trip_month"].apply(trip_month_to_dt)

    # Final filter: May 2016 to May 2026 inclusive
    df = df[
        df["_trip_month_dt"].notna() &
        (df["_trip_month_dt"] >= TRIP_START_MONTH) &
        (df["_trip_month_dt"] <= TRIP_END_MONTH)
    ].copy()

    # Deduplicate
    df["_dedup_key"] = make_dedup_key(df)
    before = len(df)
    df = df.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
    after = len(df)

    # Drop helper columns
    df = df.drop(columns=["_trip_month_dt", "_dedup_key"])

    # Optional sort to preserve scrape order
    df = df.sort_values(by=["scrape_order"], ascending=True).reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nDone.")
    print(f"Rows before dedup: {before}")
    print(f"Rows after dedup : {after}")
    print(f"Saved to         : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
