#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>

#include "horizon/SimulationTypes.hpp"
#include "physics/Relativity.cuh"

namespace {

int failures = 0;

void check(bool condition, const std::string& name, double value = 0.0, double tolerance = 0.0) {
  if (condition) {
    std::cout << "[pass] " << name;
    if (tolerance > 0.0) {
      std::cout << " value=" << value << " tol=" << tolerance;
    }
    std::cout << '\n';
    return;
  }
  std::cerr << "[fail] " << name;
  if (tolerance > 0.0) {
    std::cerr << " value=" << value << " tol=" << tolerance;
  }
  std::cerr << '\n';
  ++failures;
}

void validateCircularOrbit() {
  constexpr double mass = 1.0;
  horizon::FourVector x{0.0, 10.0, horizon::kPi * 0.5, 0.0};
  horizon::FourVector u = horizon::physics::circularOrbitFourVelocity(x.r, x.theta, mass);
  const double norm = horizon::physics::fourVelocityNorm(x, u, mass);
  const double expectedEnergy = (1.0 - 2.0 * mass / x.r) / std::sqrt(1.0 - 3.0 * mass / x.r);
  const double energy = horizon::physics::conservedEnergy(x, u, mass);
  check(std::abs(norm + 1.0) < 2.0e-14, "timelike circular orbit four-velocity normalization", norm + 1.0, 2.0e-14);
  check(std::abs(energy - expectedEnergy) < 2.0e-14, "circular orbit specific energy", energy - expectedEnergy, 2.0e-14);

  horizon::physics::GeodesicState state{x, u};
  double maxRadialDrift = 0.0;
  for (int i = 0; i < 4000; ++i) {
    state = horizon::physics::rk4Step(state, 0.0025, mass);
    maxRadialDrift = std::max(maxRadialDrift, std::abs(state.x.r - x.r));
  }
  check(maxRadialDrift < 2.0e-8, "RK4 preserves r=10M circular orbit", maxRadialDrift, 2.0e-8);
}

void validatePhotonSphere() {
  constexpr double mass = 1.0;
  const double r = 3.0 * mass;
  const double f = 1.0 - 2.0 * mass / r;
  const double impact = 3.0 * std::sqrt(3.0) * mass;

  horizon::physics::GeodesicState photon{};
  photon.x = {0.0, r, horizon::kPi * 0.5, 0.0};
  photon.u = {1.0 / f, 0.0, 0.0, impact / (r * r)};
  const double nullNorm = horizon::physics::fourVelocityNorm(photon.x, photon.u, mass);
  check(std::abs(nullNorm) < 2.0e-13, "photon sphere null normalization", nullNorm, 2.0e-13);

  double maxRadialDrift = 0.0;
  for (int i = 0; i < 8000; ++i) {
    photon = horizon::physics::rk4Step(photon, 0.00075, mass);
    maxRadialDrift = std::max(maxRadialDrift, std::abs(photon.x.r - r));
  }
  check(maxRadialDrift < 2.0e-7, "null geodesic remains on photon sphere", maxRadialDrift, 2.0e-7);
}

void validateRedshiftAndCurvature() {
  constexpr double mass = 1.0;
  const double redshift = horizon::physics::gravitationalRedshift(8.0, 1000.0, mass);
  const double expected = std::sqrt((1.0 - 2.0 / 8.0) / (1.0 - 2.0 / 1000.0));
  check(std::abs(redshift - expected) < 2.0e-15, "static gravitational redshift formula", redshift - expected, 2.0e-15);

  const double k = horizon::physics::kretschmannScalar(4.0, mass);
  const double expectedK = 48.0 / std::pow(4.0, 6.0);
  check(std::abs(k - expectedK) < 2.0e-15, "Kretschmann scalar", k - expectedK, 2.0e-15);

  const double residual = horizon::physics::vacuumEinsteinResidualAnalytic({0.0, 9.0, 1.2, 0.4}, mass);
  check(std::abs(residual) < 1.0e-15, "vacuum Einstein tensor residual outside singularity", residual, 1.0e-15);
}

void validateRelativisticPrecession() {
  constexpr double mass = 1.0;
  constexpr double p = 18.0;
  constexpr double e = 0.2;
  const double rp = p * mass / (1.0 + e);
  const double energySquared = ((p - 2.0 - 2.0 * e) * (p - 2.0 + 2.0 * e)) /
                               (p * (p - 3.0 - e * e));
  const double angularMomentumSquared = p * p * mass * mass / (p - 3.0 - e * e);

  horizon::physics::GeodesicState orbit{};
  orbit.x = {0.0, rp, horizon::kPi * 0.5, 0.0};
  orbit.u.t = std::sqrt(energySquared) / horizon::physics::metricF(rp, mass);
  orbit.u.r = 0.0;
  orbit.u.theta = 0.0;
  orbit.u.phi = std::sqrt(angularMomentumSquared) / (rp * rp);

  bool sawApapsis = false;
  double previousUr = orbit.u.r;
  double previousPhi = orbit.x.phi;
  double unwrappedPhi = 0.0;
  double periapsisAdvance = 0.0;
  for (int i = 0; i < 400000; ++i) {
    orbit = horizon::physics::rk4Step(orbit, 0.004, mass);
    double deltaPhi = orbit.x.phi - previousPhi;
    if (deltaPhi < -horizon::kPi) {
      deltaPhi += horizon::kTwoPi;
    } else if (deltaPhi > horizon::kPi) {
      deltaPhi -= horizon::kTwoPi;
    }
    unwrappedPhi += deltaPhi;
    previousPhi = orbit.x.phi;

    if (!sawApapsis && previousUr > 0.0 && orbit.u.r <= 0.0) {
      sawApapsis = true;
    } else if (sawApapsis && previousUr < 0.0 && orbit.u.r >= 0.0) {
      periapsisAdvance = unwrappedPhi;
      break;
    }
    previousUr = orbit.u.r;
  }

  const double precession = periapsisAdvance - horizon::kTwoPi;
  const double weakFieldEstimate = 6.0 * horizon::kPi * mass / p;
  check(sawApapsis, "eccentric orbit reaches apapsis");
  check(precession > 0.0, "periapsis advances in Schwarzschild geometry", precession, 0.0);
  check(std::abs(precession - weakFieldEstimate) < 0.45,
        "periapsis precession is near weak-field prediction",
        precession - weakFieldEstimate,
        0.45);
}

}  // namespace

int main() {
  validateCircularOrbit();
  validatePhotonSphere();
  validateRedshiftAndCurvature();
  validateRelativisticPrecession();

  if (failures != 0) {
    std::cerr << failures << " validation check(s) failed.\n";
    return 1;
  }

  std::cout << "All Schwarzschild validation checks passed.\n";
  return 0;
}
