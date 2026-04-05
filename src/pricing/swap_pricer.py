"""Interest rate swap pricing.

Supports plain vanilla fixed-for-floating swaps.

Conventions
-----------
* ``payment_times_fixed`` — sorted list of fixed-leg payment dates (years).
* ``payment_times_float`` — sorted list of floating-leg payment dates (years).
* Accrual periods are inferred as the difference between consecutive dates
  (with 0 as the implicit start).
* All rates and spreads are decimals (e.g. 0.03 = 3 %).
* **Swap value is from the fixed-payer perspective**:
      V_swap = PV(float leg) − PV(fixed leg)
  A positive value means the fixed payer benefits (rates have risen).
"""

from __future__ import annotations

from typing import List


class SwapPricer:
    """Price plain vanilla fixed-for-floating interest rate swaps.

    Parameters
    ----------
    None — all inputs are passed per method call.
    """

    # ------------------------------------------------------------------
    # Fixed leg
    # ------------------------------------------------------------------

    def price_fixed_leg(
        self,
        notional: float,
        fixed_rate: float,
        payment_times: List[float],
        curve,
    ) -> float:
        """Present value of the fixed leg.

        Parameters
        ----------
        notional : float
        fixed_rate : float
            Annual fixed coupon rate (decimal).
        payment_times : list of float
            Ordered coupon payment dates in years.  Accrual period is
            inferred as ``t_i − t_{i-1}`` (t_0 = 0).
        curve : InterpolatedCurve

        Returns
        -------
        float — PV of all fixed coupon cashflows (no notional exchange).
        """
        prev_t = 0.0
        pv = 0.0
        for t in payment_times:
            dt = t - prev_t
            pv += notional * fixed_rate * dt * curve.get_df(t)
            prev_t = t
        return pv

    # ------------------------------------------------------------------
    # Float leg
    # ------------------------------------------------------------------

    def price_float_leg(
        self,
        notional: float,
        payment_times: List[float],
        curve,
        spread: float = 0.0,
    ) -> float:
        """Present value of the floating leg.

        Uses the replicating portfolio identity:

            PV(float leg, spread=0) = notional * (DF(t_0) − DF(T))
                                    = notional * (1 − DF(T))   [if t_0 = 0]

        The spread contribution is the annuity of ``spread × dt_i × DF(t_i)``.

        Parameters
        ----------
        notional : float
        payment_times : list of float
        curve : InterpolatedCurve
        spread : float
            Basis spread over the floating index (decimal, default 0).

        Returns
        -------
        float — PV of the floating leg (no notional exchange).
        """
        prev_t = 0.0
        pv = 0.0
        for t in payment_times:
            dt = t - prev_t
            # Float coupon via replication: PV = notional * (DF(prev) - DF(t))
            pv += notional * (curve.get_df(prev_t) - curve.get_df(t))
            # Spread contribution
            pv += notional * spread * dt * curve.get_df(t)
            prev_t = t
        return pv

    # ------------------------------------------------------------------
    # Full swap
    # ------------------------------------------------------------------

    def price_swap(
        self,
        notional: float,
        fixed_rate: float,
        payment_times_fixed: List[float],
        payment_times_float: List[float],
        curve,
        float_spread: float = 0.0,
    ) -> float:
        """Swap NPV from the fixed-payer perspective.

            V = PV(float leg) − PV(fixed leg)

        A positive value means current market rates are above ``fixed_rate``
        (the fixed payer is in-the-money).

        Parameters
        ----------
        notional : float
        fixed_rate : float
            Annual fixed coupon rate (decimal).
        payment_times_fixed : list of float
            Fixed-leg payment dates in years.
        payment_times_float : list of float
            Floating-leg payment dates in years.
        curve : InterpolatedCurve
        float_spread : float
            Spread on the floating leg (decimal, default 0).

        Returns
        -------
        float — swap NPV (fixed payer perspective).
        """
        pv_fixed = self.price_fixed_leg(notional, fixed_rate, payment_times_fixed, curve)
        pv_float = self.price_float_leg(notional, payment_times_float, curve, spread=float_spread)
        return pv_float - pv_fixed

    # ------------------------------------------------------------------
    # Par rate (fair fixed rate)
    # ------------------------------------------------------------------

    def par_rate(
        self,
        notional: float,
        payment_times_fixed: List[float],
        payment_times_float: List[float],
        curve,
        float_spread: float = 0.0,
    ) -> float:
        """The fixed rate that makes the swap NPV equal to zero.

        Derived analytically:

            R* = [PV(float leg at R=0) + spread_annuity] / annuity_fixed

            R* = (1 − DF(T) + spread × A_float) / A_fixed

        where A = sum  dt_i × DF(t_i)  is the annuity factor.

        Parameters
        ----------
        notional : float
            Used only for scaling; drops out of the ratio.
        payment_times_fixed : list of float
        payment_times_float : list of float
        curve : InterpolatedCurve
        float_spread : float
            Spread on the floating leg (decimal, default 0).

        Returns
        -------
        float — par swap rate (annually compounded, decimal).
        """
        # Annuity (fixed leg denominator)
        annuity_fixed = sum(
            (payment_times_fixed[i] - (payment_times_fixed[i - 1] if i > 0 else 0.0))
            * curve.get_df(payment_times_fixed[i])
            for i in range(len(payment_times_fixed))
        )

        # Float leg numerator: 1 - DF(T)
        t_float_last = payment_times_float[-1]
        numerator = 1.0 - curve.get_df(t_float_last)

        # Spread annuity on float side
        if float_spread != 0.0:
            prev = 0.0
            for t in payment_times_float:
                numerator += float_spread * (t - prev) * curve.get_df(t)
                prev = t

        return numerator / annuity_fixed

    # ------------------------------------------------------------------
    # DV01 (parallel shift)
    # ------------------------------------------------------------------

    def dv01(
        self,
        notional: float,
        fixed_rate: float,
        payment_times_fixed: List[float],
        payment_times_float: List[float],
        curve,
        bump: float = 1e-4,
    ) -> float:
        """Swap DV01: change in NPV for a 1 bp parallel upward shift.

        Computed by finite difference on the discount factors.

        Returns
        -------
        float — DV01 (positive for fixed payer: benefits when rates rise).
        """
        import math

        class _BumpedCurve:
            """Wrapper that shifts rates by ±bump.

            direction=+1 means rates UP → DF * exp(-bump * t).
            direction=-1 means rates DOWN → DF * exp(+bump * t).
            """
            def __init__(self, base_curve, direction):
                self._base = base_curve
                self._sign = direction  # +1 = rates up, -1 = rates down

            def get_df(self, t):
                return self._base.get_df(t) * math.exp(-self._sign * bump * t)

        pv_up = self.price_swap(
            notional, fixed_rate, payment_times_fixed, payment_times_float,
            _BumpedCurve(curve, +1),
        )
        pv_dn = self.price_swap(
            notional, fixed_rate, payment_times_fixed, payment_times_float,
            _BumpedCurve(curve, -1),
        )
        return (pv_up - pv_dn) / 2.0
