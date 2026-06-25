"""
Probabilistic models for renewable DER output, used to drive the
Monte-Carlo probabilistic power flow. Both models output power directly
in kW so they are unit-consistent with the IEEE 33-bus load data (network.py).
"""

import numpy as np


class WindTurbine:
    """
    Wind speed ~ Weibull(k, c).  Power via standard 3-segment turbine curve.
    """

    def __init__(self, rated_kw=500.0, v_ci=3.0, v_r=12.0, v_co=25.0,
                 weibull_k=2.0, weibull_c=8.0):
        self.rated_kw = rated_kw
        self.v_ci = v_ci
        self.v_r = v_r
        self.v_co = v_co
        self.k = weibull_k
        self.c = weibull_c

    def sample_wind_speed(self, n, rng):
        return rng.weibull(self.k, size=n) * self.c

    def power_kw(self, v):
        v = np.atleast_1d(v)
        p = np.zeros_like(v, dtype=float)
        ramp = (v >= self.v_ci) & (v < self.v_r)
        rated = (v >= self.v_r) & (v <= self.v_co)
        p[ramp] = self.rated_kw * (v[ramp] - self.v_ci) / (self.v_r - self.v_ci)
        p[rated] = self.rated_kw
        return p

    def sample_power_kw(self, n, rng):
        v = self.sample_wind_speed(n, rng)
        return self.power_kw(v)


class PVSystem:
    """
    Solar irradiance G ~ Beta(alpha, beta) scaled to [0, G_max] kW/m^2.
    PV output via standard temperature-corrected model.
    """

    def __init__(self, rated_kw=400.0, g_std=1.0, g_max=1.0,
                 beta_alpha=2.5, beta_beta=1.5,
                 t_ambient=25.0, noct=45.0, temp_coeff=-0.0045, t_ref=25.0):
        self.rated_kw = rated_kw
        self.g_std = g_std          # kW/m^2 at STC
        self.g_max = g_max          # kW/m^2 max instantaneous irradiance
        self.alpha = beta_alpha
        self.beta = beta_beta
        self.t_ambient = t_ambient
        self.noct = noct
        self.temp_coeff = temp_coeff  # %/C as fraction, negative
        self.t_ref = t_ref

    def sample_irradiance(self, n, rng):
        return rng.beta(self.alpha, self.beta, size=n) * self.g_max

    def power_kw(self, g):
        g = np.atleast_1d(g)
        t_cell = self.t_ambient + g * (self.noct - 20.0) / 0.8
        p = self.rated_kw * (g / self.g_std) * (1.0 + self.temp_coeff * (t_cell - self.t_ref))
        return np.clip(p, 0.0, None)

    def sample_power_kw(self, n, rng):
        g = self.sample_irradiance(n, rng)
        return self.power_kw(g)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    wt = WindTurbine()
    pv = PVSystem()
    pw = wt.sample_power_kw(5000, rng)
    ppv = pv.sample_power_kw(5000, rng)
    print(f"Wind: mean={pw.mean():.1f} kW, max={pw.max():.1f} kW, "
          f"capacity factor={pw.mean()/wt.rated_kw:.2%}")
    print(f"PV:   mean={ppv.mean():.1f} kW, max={ppv.max():.1f} kW, "
          f"capacity factor={ppv.mean()/pv.rated_kw:.2%}")
