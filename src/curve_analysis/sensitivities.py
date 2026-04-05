"""Interest rate sensitivities: DV01, Key Rate DV01, Key Rate Duration, Partial Duration.

All sensitivities are computed by central finite difference on the zero rates
of the underlying ZC curve pillars.  The caller provides a ``price_fn(curve)``
callable that returns the mark-to-model value of any instrument or portfolio.

Key Rate DV01 uses tent (triangular) bump functions so that the sum of all
key rate buckets equals the total parallel DV01:

    For a pillar t between key tenors k_i and k_{i+1}:
      contribution from tent_i  = h * (k_{i+1} - t) / (k_{i+1} - k_i)
      contribution from tent_{i+1} = h * (t - k_i)   / (k_{i+1} - k_i)
      sum = h (parallel shift) ✓
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd

from curve_construction import InterpolatedCurve

# Default ALM-relevant key rate tenors (years)
DEFAULT_KEY_TENORS: List[float] = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

PriceFn = Callable[[InterpolatedCurve], float]


class SensitivityAnalyser:
    """Compute DV01, Key Rate Durations and Partial Durations for any instrument.

    Parameters
    ----------
    zc_df : DataFrame
        Output of ``ZCCurveBuilder``: columns ``tenor``, ``zero_rate``,
        ``discount_factor``.  The tenors become the curve pillars.
    key_tenors : list of float, optional
        Key rate tenors for bucketed sensitivity.
        Defaults to ``[1, 2, 3, 5, 7, 10, 15, 20, 25, 30]``.
    """

    def __init__(
        self,
        zc_df: pd.DataFrame,
        key_tenors: Optional[List[float]] = None,
    ) -> None:
        self._zc_df = zc_df.sort_values("tenor").reset_index(drop=True)
        self._pillar_tenors: List[float] = self._zc_df["tenor"].tolist()
        self._key_tenors: List[float] = sorted(key_tenors or DEFAULT_KEY_TENORS)
        self._base_curve = InterpolatedCurve(self._zc_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bumped_curve(self, delta_z: Dict[float, float]) -> InterpolatedCurve:
        """Rebuild an InterpolatedCurve with zero rates shifted by delta_z."""
        df = self._zc_df.copy()
        for i, row in df.iterrows():
            t = row["tenor"]
            dz = delta_z.get(t, 0.0)
            z_new = max(row["zero_rate"] + dz, -0.9999)   # keep DF > 0
            df.at[i, "zero_rate"] = z_new
            df.at[i, "discount_factor"] = 1.0 / (1.0 + z_new) ** t
        return InterpolatedCurve(df)

    def _tent_bumps(self, key_tenor: float, bump: float) -> Dict[float, float]:
        """Per-pillar bump magnitudes for the tent function at ``key_tenor``.

        The tent function ensures:
          - Full bump ``h`` at the key tenor itself.
          - Linear ramp between adjacent key tenors.
          - Flat (=h) for the leftmost tent at t ≤ k_0 or rightmost at t ≥ k_n.

        This guarantees: sum over all key rate tent bumps = parallel shift.
        """
        kt = self._key_tenors
        ki = kt.index(key_tenor)
        k_prev = kt[ki - 1] if ki > 0 else None
        k_next = kt[ki + 1] if ki < len(kt) - 1 else None

        result: Dict[float, float] = {}
        for t in self._pillar_tenors:

            if k_prev is None:
                # Leftmost key rate: flat bump for t ≤ k_i, ramp down after
                if t <= key_tenor:
                    b = bump
                elif k_next is not None and t < k_next:
                    b = bump * (k_next - t) / (k_next - key_tenor)
                else:
                    b = 0.0

            elif k_next is None:
                # Rightmost key rate: ramp up before, flat bump for t ≥ k_i
                if t >= key_tenor:
                    b = bump
                elif t > k_prev:
                    b = bump * (t - k_prev) / (key_tenor - k_prev)
                else:
                    b = 0.0

            else:
                # Middle key rate: standard tent (ramp up, ramp down)
                if k_prev <= t <= key_tenor:
                    b = bump * (t - k_prev) / (key_tenor - k_prev)
                elif key_tenor < t < k_next:
                    b = bump * (k_next - t) / (k_next - key_tenor)
                else:
                    b = 0.0

            result[t] = b
        return result

    @staticmethod
    def _label(t: float) -> str:
        return f"{int(t)}Y" if t == int(t) else f"{t}Y"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def base_curve(self) -> InterpolatedCurve:
        """The unshifted curve."""
        return self._base_curve

    def total_dv01(self, price_fn: PriceFn, bump: float = 1e-4) -> float:
        """Parallel-shift DV01: (P(+1bp) − P(−1bp)) / 2.

        Negative for a long bond (price falls when rates rise).

        Parameters
        ----------
        price_fn : callable
            ``price_fn(curve) -> float``
        bump : float
            Rate shift (default 1 bp = 0.0001).

        Returns
        -------
        float
        """
        all_up = {t: +bump for t in self._pillar_tenors}
        all_dn = {t: -bump for t in self._pillar_tenors}
        pv_up = price_fn(self._bumped_curve(all_up))
        pv_dn = price_fn(self._bumped_curve(all_dn))
        return (pv_up - pv_dn) / 2.0

    def key_rate_dv01s(
        self,
        price_fn: PriceFn,
        bump: float = 1e-4,
    ) -> Dict[str, float]:
        """DV01 per key rate bucket using tent bump functions.

        The sum of all bucket DV01s equals ``total_dv01``.

        Returns
        -------
        dict  e.g. ``{'1Y': -2.3, '2Y': -4.7, ..., '30Y': -18.1}``
        """
        result: Dict[str, float] = {}
        for kt in self._key_tenors:
            bumps_up = self._tent_bumps(kt, +bump)
            bumps_dn = self._tent_bumps(kt, -bump)
            pv_up = price_fn(self._bumped_curve(bumps_up))
            pv_dn = price_fn(self._bumped_curve(bumps_dn))
            result[self._label(kt)] = (pv_up - pv_dn) / 2.0
        return result

    def key_rate_durations(
        self,
        price_fn: PriceFn,
        price_base: Optional[float] = None,
        bump: float = 1e-4,
    ) -> Dict[str, float]:
        """Key Rate Duration (KRD) per bucket.

        KRD_i = −KR_DV01_i / (P_base × bump)

        Represents the % price change per 1 % (100 bp) shift in that bucket.
        Sum of all KRDs equals the modified duration of the instrument.

        Returns
        -------
        dict  e.g. ``{'5Y': 4.72, '10Y': 3.15, ...}``
        """
        if price_base is None:
            price_base = price_fn(self._base_curve)
        kr_dv01s = self.key_rate_dv01s(price_fn, bump)
        denom = price_base * bump if price_base != 0.0 else 1.0
        return {k: -v / denom for k, v in kr_dv01s.items()}

    def partial_durations(
        self,
        price_fn: PriceFn,
        price_base: Optional[float] = None,
        bump: float = 1e-4,
    ) -> Dict[str, float]:
        """Partial Duration per curve pillar (single-pillar bump, no tent).

        Each pillar is bumped independently:

            PD_i = −(P(z_i + h) − P(z_i − h)) / (2 × h × P_base)

        Useful for identifying exactly which maturities drive the sensitivity.
        Unlike KRDs, partial durations do not sum to modified duration
        (cross-pillar interactions are excluded).

        Returns
        -------
        dict  e.g. ``{'10Y': 8.94, '11Y': 0.01, ...}``
        """
        if price_base is None:
            price_base = price_fn(self._base_curve)
        denom = price_base * bump if price_base != 0.0 else 1.0
        result: Dict[str, float] = {}
        for t in self._pillar_tenors:
            up = {t2: (bump if abs(t2 - t) < 1e-9 else 0.0) for t2 in self._pillar_tenors}
            dn = {t2: (-bump if abs(t2 - t) < 1e-9 else 0.0) for t2 in self._pillar_tenors}
            pv_up = price_fn(self._bumped_curve(up))
            pv_dn = price_fn(self._bumped_curve(dn))
            result[self._label(t)] = -(pv_up - pv_dn) / (2.0 * denom)
        return result

    def summary(
        self,
        price_fn: PriceFn,
        bump: float = 1e-4,
    ) -> pd.DataFrame:
        """Full bucketed sensitivity report as a DataFrame.

        Columns
        -------
        * ``bucket``   – key rate label (e.g. ``'10Y'``) or ``'TOTAL'``
        * ``kr_dv01``  – key rate DV01 (monetary units per 1 bp)
        * ``krd``      – key rate duration (years)

        The ``TOTAL`` row contains the parallel DV01 and sum of KRDs.
        """
        price_base = price_fn(self._base_curve)
        total = self.total_dv01(price_fn, bump)
        kr_dv01s = self.key_rate_dv01s(price_fn, bump)
        denom = price_base * bump if price_base != 0.0 else 1.0
        krds = {k: -v / denom for k, v in kr_dv01s.items()}

        rows = [
            {"bucket": k, "kr_dv01": v, "krd": krds[k]}
            for k, v in kr_dv01s.items()
        ]
        rows.append({
            "bucket": "TOTAL",
            "kr_dv01": total,
            "krd": sum(krds.values()),
        })
        return pd.DataFrame(rows)
