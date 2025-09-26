# app.py (نسخه اصلاح‌شده — پرداختی شامل 10,000,000 و بازمحاسبه سود/زیان)
import os
import json
from io import BytesIO
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ---------------------------
# تنظیمات اولیه
# ---------------------------
load_dotenv()
DEFAULT_TIMEOUT = 15
API_KEY_DEFAULT = os.getenv("API_KEY", "")
API_URL_DEFAULT = os.getenv(
    "API_URL",
    f"https://BrsApi.ir/Api/Tsetmc/Option.php?key={API_KEY_DEFAULT}" if API_KEY_DEFAULT else ""
)
USER_AGENT_DEFAULT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

st.set_page_config(page_title="داشبورد بازار آپشن (AgGrid)", layout="wide")
st.title("📊 داشبورد بازار آپشن بورس ایران — (AgGrid)")

# ---------------------------
# توابع کمکی
# ---------------------------
def safe_get(d: dict, keys: list, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, "", " "):
            return d[k]
    return default

def to_num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        try:
            return int(x)
        except Exception:
            return None

# ---------------------------
# نرمال‌سازی آیتم‌ها (خواندن base_pc و interest_open)
# ---------------------------
def normalize_item(item: dict) -> dict:
    """Map common API keys to internal normalized keys.
       **Important changes:** we now read base_pc and interest_open if present.
    """
    symbol = safe_get(item, ["l18", "symbol", "name", "ticker", "base_l18", "base_118"])
    # price strike / exercise
    strike = safe_get(item, ["price_strike", "strike", "exercise_price", "size_contract", "price"])
    # premiums & option prices
    last_premium = safe_get(item, ["pl", "last", "last_premium", "premium"])
    option_close = safe_get(item, ["pc", "close", "close_price", "settlement_price"])
    # orderbook level1
    offer1 = safe_get(item, ["po1", "po1_price", "ask1", "ask_price", "offer1"])
    bid1 = safe_get(item, ["pd1", "pd1_price", "bid1", "bid_price", "demand1"])
    # *** Here: base_pc (correct price of underlying close) ***
    base_pc = safe_get(item, ["base_pc", "py", "base_py", "base_close", "underlying_close", "underlying_last", "price_base"])
    plp = safe_get(item, ["plp", "change_pct_last", "pl_p", "changeLastPct"])
    pcp = safe_get(item, ["pcp", "change_pct_close", "changeClosePct"])
    # *** interest_open for open interest / موقعیت باز ***
    interest_open = safe_get(item, ["interest_open", "interestOpen", "openint", "open_interest"])
    tvol = safe_get(item, ["tvol", "volume", "vol", "trade_volume"])
    tval = safe_get(item, ["tval", "value", "trade_value"])
    nval = safe_get(item, ["nval", "notional_value"])
    tno = safe_get(item, ["tno", "trades", "count", "num_trades"])

    return {
        "symbol": symbol or "N/A",
        "strike": to_num(strike),
        "last_premium": to_num(last_premium),
        "option_close": to_num(option_close),
        "offer1": to_num(offer1),
        "bid1": to_num(bid1),
        # <- use base_pc (changed per your request)
        "base_pc": to_num(base_pc),
        "plp": to_num(plp),
        "pcp": to_num(pcp),
        # <- use interest_open for position open
        "interest_open": to_num(interest_open),
        "tvol": to_num(tvol),
        "tval": to_num(tval) or to_num(nval) or 0,
        "tno": to_num(tno),
        "_raw": item
    }

def parse_api_response(resp_json):
    """Support multiple shapes of API response and return normalized items."""
    items = None
    if isinstance(resp_json, list):
        items = resp_json
    elif isinstance(resp_json, dict):
        for k in ("data", "result", "items", "contracts", "rows", "payload"):
            if k in resp_json and isinstance(resp_json[k], list):
                items = resp_json[k]
                break
        if items is None:
            for v in resp_json.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    items = v
                    break
    if items is None:
        if isinstance(resp_json, dict):
            return [normalize_item(resp_json)]
        raise ValueError("پاسخ API حاوی لیست قراردادها نبود. لطفاً ساختار JSON را بررسی کنید.")
    return [normalize_item(it) for it in items]

# ---------------------------
# فراخوانی API / بارگذاری محلی
# ---------------------------
def fetch_api(url: str, user_agent: str, timeout=DEFAULT_TIMEOUT):
    headers = {"User-Agent": user_agent}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return parse_api_response(data)

def load_local_json_file(file_like):
    data = json.load(file_like)
    return parse_api_response(data)

# ---------------------------
# توابع فرمت عدد و درصد
# ---------------------------
def format_number(x):
    try:
        if x is None:
            return ""
        if isinstance(x, (int, float)) and float(x).is_integer():
            return f"{int(x):,}"
        return f"{float(x):,}"
    except Exception:
        return str(x)

def format_optional_pct(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)

# ---------------------------
# محاسبات ستون‌های 1..15 (با base_pc و interest_open)
# ---------------------------
def build_result_df(normalized_items: list) -> pd.DataFrame:
    rows = []
    for it in normalized_items:
        symbol = it.get("symbol", "N/A")
        strike = it.get("strike") or 0.0
        # *** use base_pc (correct underlying close) ***
        base_pc = it.get("base_pc") or 0.0
        offer1 = it.get("offer1") or 0.0
        bid1 = it.get("bid1") or 0.0
        last_premium = it.get("last_premium") or 0.0
        option_close = it.get("option_close") or 0.0
        plp = it.get("plp")
        pcp = it.get("pcp")
        # *** interest_open -> موقعیت باز ***
        interest_open = it.get("interest_open") or 0
        tvol = it.get("tvol") or 0
        tval = it.get("tval") or 0

        # 2) اختلاف اعمال = ((base_pc - strike) / strike) * 100
        diff_strike_pct = None
        if strike:
            diff_strike_pct = ((base_pc - strike) / strike) * 100

        # 3) اختلاف سر به سر: break_even = base_pc + offer1
        diff_break_even_pct = None
        if base_pc:
            break_even = base_pc + offer1
            diff_break_even_pct = ((break_even - base_pc) / base_pc) * 100

        # 10) تعداد پرمیوم دریافتی
        premiums_count = 0
        if offer1 and offer1 > 0:
            premiums_count = int(10_000_000 // offer1)

        # 11) پرداختی اعمال = (premiums_count * strike) + 10_000_000   <-- تغییر اصلی
        payed_amount = premiums_count * strike

        # 4) سود و زیان (برای رشد 10% ،20% ،40%) — با استفاده از payed_amount جدید
        profit_percents = []
        for growth_factor in (1.1, 1.2, 1.4):
            if premiums_count and strike:
                grown_price = base_pc * growth_factor
                received_total = premiums_count * grown_price
                paid_total = payed_amount + 10000000 # حالا شامل 10M است
                if paid_total != 0:
                    profit_pct = ((received_total - paid_total) / paid_total) * 100
                else:
                    profit_pct = None
            else:
                profit_pct = None
            profit_percents.append(profit_pct)

        def pct_to_str(x):
            return f"{x:.2f}%" if (x is not None) else "—"

        profit_str = " | ".join([pct_to_str(p) for p in profit_percents])

        # 12) آخرین قیمت (نمایش همراه درصد تغییر) - به صورت رشته (برای نمایش درصد)
        last_price_display = f"{format_number(last_premium)}"
        if plp is not None:
            last_price_display += f" ({format_optional_pct(plp)})"

        # 13) قیمت پایانی آپشن (نمایش همراه درصد)
        option_close_display = f"{format_number(option_close)}"
        if pcp is not None:
            option_close_display += f" ({format_optional_pct(pcp)})"

        rows.append({
            "نماد": symbol,
            "اختلاف اعمال (%)": round(diff_strike_pct, 2) if diff_strike_pct is not None else None,
            "اختلاف سر به سر (%)": round(diff_break_even_pct, 2) if diff_break_even_pct is not None else None,
            "سود و زیان (10% | 20% | 40%)": profit_str,
            # 5) موقعیت باز از interest_open
            "موقعیت باز": int(interest_open) if interest_open is not None else None,
            "حجم": int(tvol) if tvol is not None else None,
            "ارزش": int(tval) if tval is not None else None,
            "قیمت عرضه": format_number(offer1),
            "قیمت تقاضا": format_number(bid1),
            "تعداد پرمیوم دریافتی": premiums_count,
            "پرداختی اعمال": int(payed_amount),
            "آخرین قیمت": last_price_display,
            "قیمت پایانی": option_close_display,
            "قیمت اعمال": format_number(strike),
            # 15) قیمت پایانی سهم -> از base_pc استفاده می‌شود
            "قیمت پایانی سهم": format_number(base_pc),
            "_raw": it.get("_raw", {}),
        })
    return pd.DataFrame(rows)

# ---------------------------
# رابط کاربری (Sidebar + Main)
# ---------------------------
with st.sidebar:
    st.header("تنظیمات دریافت داده")
    api_url = st.text_input("API URL", value=API_URL_DEFAULT)
    api_key_input = st.text_input("API Key (اختیاری)", value=API_KEY_DEFAULT)
    user_agent = st.text_input("User-Agent", value=USER_AGENT_DEFAULT)
    use_local = st.checkbox("استفاده از فایل JSON محلی برای تست", value=False)
    uploaded = None
    if use_local:
        uploaded = st.file_uploader("آپلود sample_api_response.json", type=["json"])
    st.markdown("---")
    st.header("نمایش/فیلتر")
    search_symbol = st.text_input("جستجو در نماد (قسمتی از نام)")
    st.markdown("انتخاب ستون‌ها برای نمایش (تیک بردارید تا مخفی شوند)")
    st.markdown("---")
    if st.button("🔄 دریافت/بروزرسانی داده‌ها"):
        st.session_state.get_data = True

if "get_data" not in st.session_state:
    st.session_state.get_data = False

# ---------------------------
# دریافت داده (API یا فایل محلی)
# ---------------------------
items = []
error_message = None
if st.session_state.get_data or use_local:
    try:
        with st.spinner("در حال دریافت/بارگذاری داده..."):
            if use_local:
                if uploaded:
                    normalized = load_local_json_file(uploaded)
                else:
                    try:
                        with open("sample_api_response.json", "r", encoding="utf-8") as f:
                            normalized = load_local_json_file(f)
                    except FileNotFoundError:
                        raise FileNotFoundError("فایل sample_api_response.json پیدا نشد؛ لطفاً آن را آپلود کنید یا use_local را غیرفعال کنید.")
            else:
                if not api_url:
                    raise ValueError("آدرس API معتبر وارد نشده است.")
                normalized = fetch_api(api_url, user_agent)
        items = normalized
    except Exception as e:
        error_message = str(e)

if error_message:
    st.error(f"خطا در دریافت داده: {error_message}")
    st.stop()

if not items:
    st.info("فعلاً داده‌ای بارگذاری نشده است. برای شروع در نوار کناری API را وارد کنید یا از فایل محلی استفاده کنید و سپس دکمه 'دریافت/بروزرسانی داده‌ها' را بزنید.")
    st.stop()

# ---------------------------
# ساخت جدول نهایی (محاسبات)
# ---------------------------
df_result = build_result_df(items)

# تعداد ردیف‌ها (برای اطلاع)
st.info(f"تعداد ردیف‌ها: {len(df_result)}")

# فیلترها: نماد (چند انتخابی) و نوع (از raw اگر وجود دارد)
available_symbols = df_result["نماد"].tolist()
selected_symbols = st.multiselect("انتخاب یک یا چند نماد", options=available_symbols, default=[])

# types extraction if present in raw
types = []
for r in items:
    raw = r.get("_raw", {}) if isinstance(r, dict) else {}
    t = safe_get(raw, ["type", "option_type", "kind"])
    if t:
        types.append(t)
types = sorted(list(set(types)))
selected_types = []
if types:
    selected_types = st.multiselect("فیلتر بر اساس نوع (Call/Put)", options=types, default=[])

# اعمال فیلترها
df_display = df_result.copy()
if selected_symbols:
    df_display = df_display[df_display["نماد"].isin(selected_symbols)]
if selected_types:
    mask = []
    for idx, row in df_display.iterrows():
        raw = row["_raw"] if "_raw" in row and isinstance(row["_raw"], dict) else {}
        t = safe_get(raw, ["type", "option_type", "kind"])
        mask.append(t in selected_types)
    df_display = df_display[mask]

# انتخاب ستون‌ها
all_cols = [c for c in df_display.columns if c != "_raw"]
visible_cols = st.multiselect("ستون‌های قابل نمایش (انتخاب برای نمایش)", options=all_cols, default=all_cols)
df_display = df_display[visible_cols]

# نمایش با AgGrid
gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_default_column(resizable=True, sortable=True, filter=True, suppressSizeToFit=False)
if "نماد" in df_display.columns:
    gb.configure_column("نماد", pinned="left", header_name="نماد")
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
grid_options = gb.build()

AgGrid(
    df_display,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    enable_enterprise_modules=False,
    theme="alpine",
    fit_columns_on_grid_load=True,
    height=520,
)

# ---------------------------
# دانلود Excel و CSV (اصلاح شده با BytesIO)
# ---------------------------
def get_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="options")
    buf.seek(0)
    return buf.getvalue()

def get_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

col_x, col_y = st.columns(2)
with col_x:
    excel_data = get_excel_bytes(df_display)
    st.download_button(
        label="⬇️ دانلود Excel (.xlsx)",
        data=excel_data,
        file_name=f"options_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
with col_y:
    csv_data = get_csv_bytes(df_display)
    st.download_button(
        label="⬇️ دانلود CSV",
        data=csv_data,
        file_name=f"options_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.caption("ستون 'نماد' هنگام اسکرول افقی پین شده است. از فیلترها و انتخاب ستون‌ها برای سفارشی‌سازی نمایش استفاده کنید.")
