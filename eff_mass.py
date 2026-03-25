import numpy as np

# -----------------------------
# Hard-coded input values
# -----------------------------

# Fractional k-points around the band edge (Quantum ESPRESSO format)
k0 = np.array([0.5, 0.5, 0.5])        # R point
k1 = np.array([0.49, 0.49, 0.49])     # Neighbor along R -> Γ

# Energies at these k-points (in eV)
E0 = 3.451
E1 = 3.4569

# Lattice constant (in Å). MODIFY FOR YOUR SYSTEM.
# Used to convert fractional k-distance to absolute |k| in m⁻¹.
a = 5.8699  # example lattice constant


# -----------------------------
# Constants
# -----------------------------
hbar = 1.054571817e-34       # J·s
eV_to_J = 1.602176634e-19     # J / eV
m_e = 9.1093837015e-31        # kg


# -----------------------------
# Convert fractional Δk to |Δk| in m^-1
# -----------------------------
# |k| = frac * |b|, with |b| = 2π/a  (for cubic lattice)

b = 2 * np.pi / (a * 1e-10)   # reciprocal lattice vector magnitude (m^-1)
delta_k_frac = np.linalg.norm(k1 - k0)
delta_k = delta_k_frac * b    # absolute Δk in m^-1

# -----------------------------
# Compute second derivative
# -----------------------------
# E1 - E0 = 1/2 E'' (Δk)^2  => E'' = 2 ΔE / (Δk)^2

delta_E = (E1 - E0) * eV_to_J  # convert to J
E_second = 2 * delta_E / (delta_k ** 2)

# -----------------------------
# Effective mass
# -----------------------------
m_eff = hbar**2 / E_second
m_eff_me = m_eff / m_e

# -----------------------------
# Print results
# -----------------------------
print("Δk fractional =", delta_k_frac)
print("Δk (absolute) =", delta_k, "m^-1")
print("ΔE =", delta_E, "J")
print("Second derivative E'' =", E_second, "J·m^2")
print("Effective mass =", m_eff, "kg")
print("Effective mass (in m_e) =", m_eff_me)
