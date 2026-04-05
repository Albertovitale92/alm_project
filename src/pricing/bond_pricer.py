"""Bond pricing using a discount curve.

Supported instruments
---------------------
* Fixed-rate bullet bond
* Fixed-rate amortising bond (arbitrary schedule)
* Floating rate note (FRN)
* Generic cashflow stream

All prices are expressed as a monetary amount (same units as *face* / *notional*).
Divide by face to get the clean/dirty price per 100.
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# A cashflow is a (time_in_years, amount) pair.
Cashflow = Tuple[float, float]


class BondPricer:
    """Price fixed-income instruments on a given discount curve.

    All methods are stateless — pass the curve explicitly so the same pricer
    instance can be reused with different curve scenarios (stress tests, etc.).

    Parameters
    ----------
    None — all inputs are passed per method call.
    """

    # ------------------------------------------------------------------
    # Core: generic cashflow pricer
    # ------------------------------------------------------------------

    def price_cashflows(
        self,
        cashflows: List[Cashflow],
        curve,
    ) -> float:
        """Present value of an arbitrary stream of cashflows.

        Parameters
        ----------
        cashflows : list of (t, amount)
            Each element is a ``(time_in_years, cash_amount)`` tuple.
        curve : InterpolatedCurve
            Provides :meth:`get_df`.

        Returns
        -------
        float — sum of DF(t_i) * amount_i
        """
        return sum(curve.get_df(t) * cf for t, cf in cashflows)

    # ------------------------------------------------------------------
    # Fixed-rate bullet bond
    # ------------------------------------------------------------------

    def price_bullet(
        self,
        face: float,
        coupon_rate: float,
        maturity: float,
        frequency: int,
        curve,
    ) -> float:
        """Price a standard fixed-rate bullet bond.

        Parameters
        ----------
        face : float
            Notional / face value.
        coupon_rate : float
            Annual coupon rate as a decimal (e.g. 0.03 for 3 %).
        maturity : float
            Bond maturity in years.
        frequency : int
            Coupon payments per year (1 = annual, 2 = semi-annual, 4 = quarterly).
        curve : InterpolatedCurve

        Returns
        -------
        float — full (dirty) price.
        """
        dt = 1.0 / frequency
        n = round(maturity * frequency)
        coupon = face * coupon_rate / frequency

        cashflows: List[Cashflow] = [
            ((i + 1) * dt, coupon) for i in range(n - 1)
        ]
        # Last payment: coupon + face redemption
        cashflows.append((n * dt, coupon + face))

        return self.price_cashflows(cashflows, curve)

    # ------------------------------------------------------------------
    # Fixed-rate amortising bond
    # ------------------------------------------------------------------

    def price_amortizing(
        self,
        schedule: List[dict],
        curve,
    ) -> float:
        """Price a fixed-rate amortising bond from an explicit schedule.

        Parameters
        ----------
        schedule : list of dicts
            Each dict must contain:

            * ``time``      – payment date in years (float)
            * ``coupon``    – coupon cash amount at this date (float, default 0)
            * ``principal`` – principal repayment at this date (float, default 0)

        curve : InterpolatedCurve

        Returns
        -------
        float — present value of all scheduled cashflows.
        """
        cashflows: List[Cashflow] = [
            (item["time"], item.get("coupon", 0.0) + item.get("principal", 0.0))
            for item in schedule
        ]
        return self.price_cashflows(cashflows, curve)

    # ------------------------------------------------------------------
    # Floating rate note
    # ------------------------------------------------------------------

    def price_frn(
        self,
        face: float,
        spread: float,
        payment_times: List[float],
        curve,
    ) -> float:
        """Price a floating rate note (FRN).

        The FRN resets at the start of each period and pays at the end:

            coupon_i = face * (L(t_{i-1}, t_i) + spread) * (t_i - t_{i-1})

        where ``L`` is the simply compounded forward rate.  Using the
        replicating portfolio identity:

            PV(float leg, spread=0) = face * DF(t_0)   [= face if just reset]

        This correctly prices to par when spread = 0 and t_0 = 0 (just-reset).

        Parameters
        ----------
        face : float
            Notional.
        spread : float
            Annual spread over floating index, as a decimal (e.g. 0.005 for 50 bp).
        payment_times : list of float
            Ordered list of coupon payment dates in years.  The first reset is
            assumed today (t=0); the last payment also returns the face value.
        curve : InterpolatedCurve

        Returns
        -------
        float — present value of the FRN (float leg + face repayment).
        """
        prev_t = 0.0
        pv = 0.0

        for t in payment_times:
            dt = t - prev_t
            # Simply compounded forward: L = (DF(prev)/DF(t) - 1) / dt
            # PV of float coupon = face * (DF(prev) - DF(t))
            pv += face * (curve.get_df(prev_t) - curve.get_df(t))
            # PV of spread coupon
            pv += face * spread * dt * curve.get_df(t)
            prev_t = t

        # Face redemption at maturity
        pv += face * curve.get_df(payment_times[-1])
        return pv

    # ------------------------------------------------------------------
    # Yield to maturity
    # ------------------------------------------------------------------

    def ytm(
        self,
        price: float,
        cashflows: List[Cashflow],
        guess: float = 0.05,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> float:
        """Yield to maturity via Newton-Raphson.

        Finds the flat rate ``y`` such that:

            price = sum  CF_i / (1 + y) ^ t_i

        Parameters
        ----------
        price : float
            Observed (dirty) price.
        cashflows : list of (t, amount)
        guess : float
            Initial yield estimate (default 5 %).
        tol : float
            Convergence tolerance on the yield change.
        max_iter : int
            Maximum Newton-Raphson iterations.

        Returns
        -------
        float — yield to maturity (annually compounded, decimal).

        Raises
        ------
        RuntimeError if the algorithm does not converge.
        """
        y = guess
        for _ in range(max_iter):
            pv = sum(cf / (1.0 + y) ** t for t, cf in cashflows)
            dpv = sum(-t * cf / (1.0 + y) ** (t + 1.0) for t, cf in cashflows)
            delta = (pv - price) / dpv
            y -= delta
            if abs(delta) < tol:
                return y
        raise RuntimeError(
            f"YTM did not converge after {max_iter} iterations "
            f"(last delta={delta:.2e})"
        )

    # ------------------------------------------------------------------
    # Duration & DV01
    # ------------------------------------------------------------------

    def modified_duration(
        self,
        cashflows: List[Cashflow],
        y: float,
    ) -> float:
        """Modified duration at yield ``y``.

        MD = (1 / P) * sum  t_i * CF_i / (1 + y) ^ (t_i + 1)
        """
        pv = sum(cf / (1.0 + y) ** t for t, cf in cashflows)
        duration = sum(t * cf / (1.0 + y) ** (t + 1.0) for t, cf in cashflows)
        return duration / pv

    def dv01(
        self,
        cashflows: List[Cashflow],
        curve,
        bump: float = 1e-4,
    ) -> float:
        """Dollar value of 1 basis point (parallel shift of the curve).

        Computed by finite difference: (P(curve + 1bp) - P(curve - 1bp)) / 2.

        Parameters
        ----------
        cashflows : list of (t, amount)
        curve : InterpolatedCurve
        bump : float
            Shift size in rate terms (default 1bp = 0.0001).

        Returns
        -------
        float — price change per 1 bp shift (negative for long bond positions).
        """
        pv_up = sum(
            curve.get_df(t) * math.exp(-bump * t) * cf for t, cf in cashflows
        )
        pv_dn = sum(
            curve.get_df(t) * math.exp(+bump * t) * cf for t, cf in cashflows
        )
        return (pv_up - pv_dn) / 2.0


# ---------------------------------------------------------------------------
# math import needed by dv01
# ---------------------------------------------------------------------------
import math  # noqa: E402  (placed here to keep the class definition clean)
