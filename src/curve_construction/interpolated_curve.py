import math

import numpy as np
import pandas as pd


class InterpolatedCurve:
    """Continuous yield curve via log-linear interpolation on discount factors.

    ``ln(DF(t))`` is interpolated linearly in ``t``.  This is equivalent to
    linear interpolation of continuously compounded zero rates and is the
    industry-standard approach in risk management because it:

    * guarantees non-negative instantaneous forward rates between pillars,
    * preserves the no-arbitrage property of the discount curve,
    * is straightforward to implement without external solvers.

    **Extrapolation**: beyond the curve endpoints the continuously compounded
    zero rate is held flat (i.e. the last observed cc rate is applied to any
    tenor outside the pillar range).

    Parameters
    ----------
    zc_df : DataFrame
        Output of :class:`~curve_construction.zc_curve_builder.ZCCurveBuilder`,
        with columns ``tenor``, ``zero_rate``, ``discount_factor``.
    """

    def __init__(self, zc_df: pd.DataFrame) -> None:
        zc_df = zc_df.sort_values("tenor").reset_index(drop=True)
        self._tenors: np.ndarray = zc_df["tenor"].to_numpy(dtype=float)
        self._log_dfs: np.ndarray = np.log(
            zc_df["discount_factor"].to_numpy(dtype=float)
        )
        # Continuously compounded zero rates at each pillar: r_c = -ln(DF)/t
        self._r_cc: np.ndarray = -self._log_dfs / self._tenors

    # ------------------------------------------------------------------
    # Internal: log(DF) at arbitrary t
    # ------------------------------------------------------------------

    def _log_df_at(self, t: float) -> float:
        """ln(DF(t)) via log-linear interpolation with flat-cc extrapolation."""
        if t == 0.0:
            return 0.0

        tenors = self._tenors
        log_dfs = self._log_dfs
        r_cc = self._r_cc

        if t < tenors[0]:
            # flat cc rate at short end: ln(DF) = -r_cc[0] * t
            return -r_cc[0] * t

        if t > tenors[-1]:
            # flat cc rate at long end: ln(DF) = -r_cc[-1] * t
            return -r_cc[-1] * t

        # Linear interpolation of ln(DF) between surrounding pillars
        # np.interp does exactly this and handles the boundary correctly
        return float(np.interp(t, tenors, log_dfs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_df(self, t: float) -> float:
        """Discount factor DF(t).

        DF(0) = 1 by definition.
        """
        if t == 0.0:
            return 1.0
        return math.exp(self._log_df_at(t))

    def get_zero_rate(self, t: float, continuous: bool = False) -> float:
        """Zero rate at tenor ``t``.

        Parameters
        ----------
        t : float
            Maturity in years.
        continuous : bool
            If ``True``, return the continuously compounded rate.
            If ``False`` (default), return the annually compounded rate,
            consistent with the ECB input convention.
        """
        if t == 0.0:
            return 0.0
        r_c = -self._log_df_at(t) / t          # continuously compounded
        if continuous:
            return r_c
        return math.exp(r_c) - 1.0             # annual compounding

    def get_simply_compounded_fwd(self, t1: float, t2: float) -> float:
        """Simply compounded forward rate for the period [t1, t2].

        Defined as:

            L(t1, t2) = (DF(t1) / DF(t2) - 1) / (t2 - t1)

        This is the rate used in FRN pricing because it satisfies the
        no-arbitrage replication: price of FRN at par when spread = 0.
        """
        if t1 >= t2:
            raise ValueError(f"t1 ({t1}) must be < t2 ({t2})")
        df1 = self.get_df(t1)
        df2 = self.get_df(t2)
        return (df1 / df2 - 1.0) / (t2 - t1)

    def get_forward_rate(self, t1: float, t2: float, continuous: bool = False) -> float:
        """Fair forward rate for the period [t1, t2].

        Derived from the no-arbitrage relation DF(t1, t2) = DF(t2) / DF(t1).

        Parameters
        ----------
        t1, t2 : float
            Start and end of the forward period in years.  ``t1 < t2``.
        continuous : bool
            If ``True``, return continuously compounded rate.
            If ``False`` (default), return annually compounded rate.
        """
        if t1 >= t2:
            raise ValueError(f"t1 ({t1}) must be < t2 ({t2})")
        if t1 == 0.0:
            return self.get_zero_rate(t2, continuous=continuous)
        l1 = self._log_df_at(t1)
        l2 = self._log_df_at(t2)
        r_c = (l1 - l2) / (t2 - t1)            # continuously compounded fwd
        if continuous:
            return r_c
        return math.exp(r_c) - 1.0             # annual compounding

    # ------------------------------------------------------------------
    # Convenience: annuity factor
    # ------------------------------------------------------------------

    def annuity_factor(self, payment_times: list) -> float:
        """Sum of discount factors: A = sum DF(t_i) * dt_i.

        Useful for swap par rate computation.  ``payment_times`` is the list
        of coupon dates; ``dt_i`` is inferred as the accrual period
        (difference between consecutive dates, with 0 as the initial point).
        """
        prev = 0.0
        total = 0.0
        for t in payment_times:
            dt = t - prev
            total += dt * self.get_df(t)
            prev = t
        return total
