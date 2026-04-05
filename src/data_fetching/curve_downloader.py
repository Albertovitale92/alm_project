import io
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from urllib.parse import urlparse
import json

import pandas as pd
import requests
from pydantic import BaseModel, ValidationError

# ============================================================
# LOGGING AVANZATO
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = TimedRotatingFileHandler(
    LOG_FILE,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)


# ============================================================
# MODELLO VALIDAZIONE
# ============================================================

class CurvePoint(BaseModel):
    tenor: str
    rate: float


# ============================================================
# NORMALIZZAZIONE COLONNE
# ============================================================

def normalize_columns(df):
    """
    Converte qualsiasi formato (ECB raw, sample, cached validato)
    nel formato standard: MATURITY | OBS_VALUE
    """
    df.columns = [c.strip().upper() for c in df.columns]

    column_map = {
        # raw ECB
        "MATURITY": "MATURITY",
        "OBS_VALUE": "OBS_VALUE",

        # file validati salvati ieri
        "TENOR": "MATURITY",
        "RATE": "OBS_VALUE",
    }

    df = df.rename(columns=column_map)
    return df


# ============================================================
# LETTURA EXCEL
# ============================================================

def read_excel_smart(path):
    if path.lower().endswith((".csv", ".txt")):
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_excel(path)
    return normalize_columns(df)


# ============================================================
# DOWNLOADER
# ============================================================

class CurveDownloader:
    MIN_REQUIRED_TENORS = 6

    # Alternative sources (future reference):
    # ECB SDW: https://sdw-wsrest.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM?format=csvdata
    # Nasdaq ECB/YCC: https://data.nasdaq.com/api/v3/datasets/ECB/YCC.json (needs NASDAQ_DATA_LINK_API_KEY)
    # FRED: IRLTLT01EZM156N (10Y), IR3TIB01EZM156N (3M), IRSTCI01EZM156N (ON) — needs FRED_API_KEY
    # GitHub sample: https://raw.githubusercontent.com/Albertovitale92/alm_project/main/data/ecb_curve_sample.csv

    EUROSTAT_URLS = [
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/IRT_EURYLD_M",
    ]

    PROXIES = [
        {"https": "http://51.158.68.133:8811"},
        {"https": "http://51.159.115.233:3128"},
        {"https": "http://195.154.43.86:80"},
    ]

    def __init__(self, save_path="data", fallback_file="ecb_curve_sample.xlsx", source_urls=None):
        self.save_path = save_path
        self.fallback_file = fallback_file
        os.makedirs(save_path, exist_ok=True)

        env_urls = os.getenv("CURVE_SOURCE_URLS")
        if env_urls:
            self.source_urls = [u.strip() for u in env_urls.split(";") if u.strip()]
        else:
            self.source_urls = source_urls or list(self.EUROSTAT_URLS)

    def _build_eurostat_curve(self, data):
        """Decode Eurostat JSON-stat dataset into MATURITY/OBS_VALUE for latest time point."""
        dim_order = data.get("id", [])
        sizes = data.get("size", [])
        dimensions = data.get("dimension", {})
        values = data.get("value", {})

        if not dim_order or not sizes or not dimensions:
            raise ValueError("Invalid Eurostat JSON-stat payload")

        if "maturity" not in dim_order or "time" not in dim_order:
            raise ValueError("Eurostat dataset missing maturity/time dimensions")

        selected_idx = {}
        for dim in dim_order:
            cat_index = dimensions[dim]["category"].get("index", {})
            if not cat_index:
                raise ValueError(f"Eurostat dimension {dim} has no categories")

            # Choose preferred categories where relevant.
            if dim == "yld_curv":
                selected_idx[dim] = cat_index.get("SPOT_RT", next(iter(cat_index.values())))
            elif dim == "bonds":
                selected_idx[dim] = cat_index.get("CGB_EA", next(iter(cat_index.values())))
            elif dim == "geo":
                selected_idx[dim] = cat_index.get("EA", next(iter(cat_index.values())))
            elif dim == "time":
                selected_idx[dim] = max(cat_index.values())
            else:
                selected_idx[dim] = next(iter(cat_index.values()))

        maturity_index = dimensions["maturity"]["category"].get("index", {})
        maturity_labels = dimensions["maturity"]["category"].get("label", {})

        def flat_index(index_map):
            idx = 0
            for i, dim in enumerate(dim_order):
                stride = 1
                for s in sizes[i + 1:]:
                    stride *= s
                idx += index_map[dim] * stride
            return idx

        rows = []
        for maturity_code, maturity_pos in sorted(maturity_index.items(), key=lambda kv: kv[1]):
            idx_map = dict(selected_idx)
            idx_map["maturity"] = maturity_pos
            obs_key = str(flat_index(idx_map))
            obs_value = values.get(obs_key)
            if obs_value is None:
                continue

            label = maturity_labels.get(maturity_code, maturity_code)
            tenor = maturity_code.replace("Y", "Y").replace("M", "M")
            if maturity_code.startswith("Y") and maturity_code[1:].isdigit():
                tenor = maturity_code[1:] + "Y"
            elif maturity_code.startswith("M") and maturity_code[1:].isdigit():
                tenor = maturity_code[1:] + "M"
            else:
                tenor = str(label)

            rows.append({"MATURITY": tenor, "OBS_VALUE": float(obs_value)})

        if not rows:
            raise ValueError("No Eurostat observations found for selected curve")

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # CACHING
    # --------------------------------------------------------
    def get_cached_curve(self):
        today = datetime.today().strftime("%Y-%m-%d")
        today_file = os.path.join(self.save_path, f"ecb_curve_{today}.xlsx")

        # File di oggi
        if os.path.exists(today_file):
            try:
                df = read_excel_smart(today_file)
                if df.shape[0] > 0:
                    logger.info(f"Using cached curve for today: {today_file}")
                    return df
            except Exception as e:
                logger.warning(f"Invalid cached file {today_file}: {e}")

        # File più recente
        files = [
            f for f in os.listdir(self.save_path)
            if f.startswith("ecb_curve_") and f.endswith(".xlsx")
        ]
        files.sort(reverse=True)

        for f in files:
            path = os.path.join(self.save_path, f)
            try:
                df = read_excel_smart(path)
                if df.shape[0] > 0:
                    logger.info(f"Using most recent cached curve: {path}")
                    return df
            except Exception:
                continue

        return None

    def try_download(self, url, proxies=None):
        try:
            parsed = urlparse(url)
            if parsed.scheme in {"http", "https"}:
                response = requests.get(url, timeout=3, proxies=proxies)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "").lower()
                if url.lower().endswith((".xls", ".xlsx")) or content_type.startswith("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
                    df = pd.read_excel(io.BytesIO(response.content))
                elif content_type.startswith("application/json") or "api.stlouisfed.org" in url or "data.nasdaq.com" in url:
                    data = response.json()
                    if data.get("class") == "dataset" and "dimension" in data and "id" in data:
                        # Eurostat JSON-stat format
                        df = self._build_eurostat_curve(data)
                    elif "dataset" in data and isinstance(data["dataset"], dict):
                        # Nasdaq Data Link / Quandl format for ECB YCC full curve
                        dataset = data["dataset"]
                        columns = dataset.get("column_names", [])
                        rows = dataset.get("data", [])
                        if not rows:
                            raise ValueError("No rows in Nasdaq Data Link response")
                        latest = rows[0]
                        df = pd.DataFrame([dict(zip(columns, latest))])
                        if "Date" in df.columns:
                            df = df.melt(id_vars=["Date"], var_name="MATURITY", value_name="OBS_VALUE")
                            df = df[df["MATURITY"] != "Date"]
                            df["MATURITY"] = df["MATURITY"].astype(str).str.strip().str.upper()
                            df = df[["MATURITY", "OBS_VALUE"]]
                        else:
                            raise ValueError("Unexpected Nasdaq Data Link format: missing Date column")
                    else:
                        # FRED API JSON response (single tenor)
                        observations = data.get("observations", [])
                        if observations:
                            latest = observations[-1]
                            value = float(latest.get("value", 0))
                            df = pd.DataFrame({"MATURITY": ["10Y"], "OBS_VALUE": [value]})
                        else:
                            raise ValueError("No observations in FRED response")
                else:
                    df = pd.read_csv(io.StringIO(response.text), sep=None, engine="python")
            elif os.path.exists(url) or os.path.exists(os.path.join(self.save_path, url)):
                local_path = url if os.path.exists(url) else os.path.join(self.save_path, url)
                if local_path.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(local_path)
                else:
                    df = pd.read_csv(local_path, sep=None, engine="python")
            else:
                raise ValueError(f"Unknown source type or path does not exist: {url}")

            df = normalize_columns(df)
            return df

        except Exception as e:
            if "NameResolutionError" in str(e):
                logger.warning(
                    f"Download failed due to DNS resolution error (proxy={proxies}): {e}"
                )
            elif isinstance(e, requests.exceptions.ProxyError):
                logger.warning(
                    f"Download failed due to proxy error (proxy={proxies}): {e}"
                )
            elif isinstance(e, requests.exceptions.ConnectionError):
                logger.warning(
                    f"Download failed due to connection error (proxy={proxies}): {e}"
                )
            elif isinstance(e, requests.exceptions.Timeout):
                logger.warning(
                    f"Download failed due to timeout after 3s (proxy={proxies}): {e}"
                )
            else:
                logger.warning(f"Download failed (proxy={proxies}): {e}")
            return None

    def download_ecb_curve(self):
        logger.info("Attempting to download ECB yield curve...")
        logger.info(f"Trying source URLs in order: {self.source_urls}")

        def is_sufficient_curve(df):
            if df is None or df.shape[0] == 0:
                return False
            required = {"MATURITY", "OBS_VALUE"}
            if not required.issubset(df.columns):
                return False
            return df.shape[0] >= self.MIN_REQUIRED_TENORS

        best_partial_df = None
        best_partial_source = None

        # Try remote URLs (and special markers) without proxy first.
        remote_or_marker_urls = [
            u for u in self.source_urls
            if isinstance(u, str) and (u.startswith("http") or u == "__FRED_MULTI__")
        ]
        for url in remote_or_marker_urls:
            df = self.try_download(url)
            if df is not None:
                if is_sufficient_curve(df):
                    logger.info(f"Download successful without proxy from {url}.")
                    return df
                if best_partial_df is None or df.shape[0] > best_partial_df.shape[0]:
                    best_partial_df = df
                    best_partial_source = url
                logger.warning(
                    f"Source {url} returned partial curve ({df.shape[0]} tenors), continuing search for fuller curve."
                )

        # Try with proxies (remote URLs only)
        remote_urls = [u for u in remote_or_marker_urls if u.startswith("http")]
        for proxy in self.PROXIES:
            for url in remote_urls:
                df = self.try_download(url, proxies=proxy)
                if df is not None:
                    if is_sufficient_curve(df):
                        logger.info(f"Download successful using proxy {proxy} from {url}.")
                        return df
                    if best_partial_df is None or df.shape[0] > best_partial_df.shape[0]:
                        best_partial_df = df
                        best_partial_source = f"{url} via proxy {proxy}"
                    logger.warning(
                        f"Source {url} via proxy returned partial curve ({df.shape[0]} tenors), continuing search."
                    )

        # Local fallback files are final options; CSV intentionally last.
        fallback_files = [
            os.path.join(self.save_path, self.fallback_file),
            os.path.join(self.save_path, "ecb_curve_sample.xlsx"),
            os.path.join(self.save_path, "ecb_curve_sample.csv"),
        ]
        for fallback_path in fallback_files:
            if os.path.exists(fallback_path):
                logger.error(
                    f"All remote sources failed. Loading local fallback curve from {fallback_path}."
                )
                return read_excel_smart(fallback_path)

        if best_partial_df is not None:
            logger.warning(
                f"No full curve found. Using best available partial curve from {best_partial_source} ({best_partial_df.shape[0]} tenors)."
            )
            return best_partial_df

        # If everything failed
        logger.error("All sources exhausted. No valid curve data available.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # VALIDAZIONE
    # --------------------------------------------------------
    def validate_curve(self, df):
        required = {"MATURITY", "OBS_VALUE"}
        if not required.issubset(df.columns):
            logger.error(f"Invalid curve file: missing columns {required - set(df.columns)}")
            return pd.DataFrame()

        validated = []
        for _, row in df.iterrows():
            try:
                cp = CurvePoint(
                    tenor=row["MATURITY"],
                    rate=row["OBS_VALUE"]
                )
                validated.append(cp.dict())
            except ValidationError as e:
                logger.warning(f"Invalid row skipped: {e}")

        return pd.DataFrame(validated)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    def save_curve(self, df):
        if df is None or df.shape[0] == 0:
            logger.error("Attempted to save an empty or invalid curve. Aborting save.")
            return None

        today = datetime.today().strftime("%Y-%m-%d")
        filename = f"ecb_curve_{today}.xlsx"
        path = os.path.join(self.save_path, filename)

        df.to_excel(path, index=False)
        logger.info(f"Curve saved to: {path}")
        return path

    # --------------------------------------------------------
    # MAIN
    # --------------------------------------------------------
    def run(self):
        df_raw = self.download_ecb_curve()
        df_valid = self.validate_curve(df_raw)

        if df_valid is not None and df_valid.shape[0] > 0:
            return self.save_curve(df_valid)

        logger.info("Downloaded curve invalid or unavailable. Trying cached curve.")
        cached = self.get_cached_curve()
        if cached is not None:
            df_valid = self.validate_curve(cached)
            return self.save_curve(df_valid)

        logger.error("No valid ECB curve available from download or cache.")
        return None