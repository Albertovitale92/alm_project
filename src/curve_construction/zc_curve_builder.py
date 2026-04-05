import re

import pandas as pd


def _parse_tenor(tenor_str: str) -> float:
    """Convert a tenor string to years (float).

    Examples: '1Y' -> 1.0, '6M' -> 0.5, '30Y' -> 30.0, '3' -> 3.0
    """
    s = str(tenor_str).strip().upper()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(Y|M)", s)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        return val if unit == "Y" else val / 12.0
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Cannot parse tenor: {tenor_str!r}")


class ZCCurveBuilder:
    """Build a zero-coupon curve from a DataFrame of ECB spot rates.

    The ECB IRT_EURYLD_M dataset provides annually compounded zero-coupon
    (spot) rates in percentage terms for maturities 1Y–30Y.  This class
    converts them into discount factors using the standard relation:

        DF(t) = 1 / (1 + z(t)) ^ t

    Parameters
    ----------
    None — call :meth:`build` with the ECB DataFrame.
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct the ZC curve.

        Parameters
        ----------
        df : DataFrame
            Must contain columns ``MATURITY`` (e.g. ``'1Y'``) and
            ``OBS_VALUE`` (annually compounded spot rate in %).

        Returns
        -------
        DataFrame with columns:

        * ``tenor``           – maturity in years (float)
        * ``zero_rate``       – annually compounded zero rate (decimal, not %)
        * ``discount_factor`` – DF(t) = 1 / (1 + z) ^ t
        """
        rows = []
        for _, row in df.iterrows():
            t = _parse_tenor(row["MATURITY"])
            z = float(row["OBS_VALUE"]) / 100.0          # % → decimal
            df_val = 1.0 / (1.0 + z) ** t
            rows.append({
                "tenor": t,
                "zero_rate": z,
                "discount_factor": df_val,
            })

        result = (
            pd.DataFrame(rows)
            .sort_values("tenor")
            .reset_index(drop=True)
        )
        return result
