#pragma once

#include <cmath>
#include <cstdint>

#include "horizon/SimulationTypes.hpp"
#include "physics/DoubleDouble.cuh"

namespace horizon::physics {

struct GeodesicState {
  FourVector x;
  FourVector u;
};

struct Cartesian {
  double x;
  double y;
  double z;
};

HORIZON_HD inline double clamp(double value, double lo, double hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

HORIZON_HD inline double absd(double value) {
  return value < 0.0 ? -value : value;
}

HORIZON_HD inline double safeSin(double theta) {
  const double s = sin(theta);
  return absd(s) < 1.0e-8 ? (s < 0.0 ? -1.0e-8 : 1.0e-8) : s;
}

HORIZON_HD inline double metricF(double r, double mass) {
  const double rs = 2.0 * mass;
  const double sep = horizonSeparation(r, rs);
  return clamp(sep / r, 1.0e-12, 1.0e12);
}

HORIZON_HD inline SchwarzschildMetric metric(const FourVector& x, double mass) {
  const double f = metricF(x.r, mass);
  const double s = safeSin(x.theta);
  return {-f, 1.0 / f, x.r * x.r, x.r * x.r * s * s};
}

HORIZON_HD inline double fourVelocityNorm(const FourVector& x, const FourVector& u, double mass) {
  const SchwarzschildMetric g = metric(x, mass);
  return g.g_tt * u.t * u.t + g.g_rr * u.r * u.r +
         g.g_thth * u.theta * u.theta + g.g_phph * u.phi * u.phi;
}

HORIZON_HD inline double conservedEnergy(const FourVector& x, const FourVector& u, double mass) {
  return metricF(x.r, mass) * u.t;
}

HORIZON_HD inline double conservedAngularMomentumZ(const FourVector& x, const FourVector& u) {
  const double s = safeSin(x.theta);
  return x.r * x.r * s * s * u.phi;
}

HORIZON_HD inline double gravitationalDilation(const FourVector& x, double mass) {
  return sqrt(metricF(x.r, mass));
}

HORIZON_HD inline double kretschmannScalar(double r, double mass) {
  return 48.0 * mass * mass / (r * r * r * r * r * r);
}

HORIZON_HD inline double radialTidalEigenvalue(double r, double mass) {
  return 2.0 * mass / (r * r * r);
}

HORIZON_HD inline Cartesian sphericalToCartesian(const FourVector& x) {
  const double s = sin(x.theta);
  return {x.r * s * cos(x.phi), x.r * cos(x.theta), x.r * s * sin(x.phi)};
}

HORIZON_HD inline FourVector add(FourVector a, FourVector b) {
  return {a.t + b.t, a.r + b.r, a.theta + b.theta, a.phi + b.phi};
}

HORIZON_HD inline FourVector scale(FourVector a, double s) {
  return {a.t * s, a.r * s, a.theta * s, a.phi * s};
}

HORIZON_HD inline FourVector addScaled(FourVector a, FourVector b, double s) {
  return {a.t + b.t * s, a.r + b.r * s, a.theta + b.theta * s, a.phi + b.phi * s};
}

HORIZON_HD inline FourVector coordinateAcceleration(const FourVector& x, const FourVector& u, double mass) {
  const double r = x.r;
  const double theta = clamp(x.theta, 1.0e-7, kPi - 1.0e-7);
  const double s = safeSin(theta);
  const double c = cos(theta);
  const double f = metricF(r, mass);

  const double gamma_t_tr = mass / (r * r * f);
  const double gamma_r_tt = mass * f / (r * r);
  const double gamma_r_rr = -mass / (r * r * f);
  const double gamma_r_thth = -r * f;
  const double gamma_r_phph = -r * f * s * s;
  const double gamma_th_rth = 1.0 / r;
  const double gamma_th_phph = -s * c;
  const double gamma_ph_rph = 1.0 / r;
  const double gamma_ph_thph = c / s;

  FourVector a{};
  a.t = -2.0 * gamma_t_tr * u.t * u.r;
  a.r = -(gamma_r_tt * u.t * u.t + gamma_r_rr * u.r * u.r +
          gamma_r_thth * u.theta * u.theta + gamma_r_phph * u.phi * u.phi);
  a.theta = -(2.0 * gamma_th_rth * u.r * u.theta + gamma_th_phph * u.phi * u.phi);
  a.phi = -(2.0 * gamma_ph_rph * u.r * u.phi + 2.0 * gamma_ph_thph * u.theta * u.phi);
  return a;
}

HORIZON_HD inline GeodesicState geodesicDerivative(const GeodesicState& state, double mass) {
  return {state.u, coordinateAcceleration(state.x, state.u, mass)};
}

HORIZON_HD inline GeodesicState addScaled(const GeodesicState& state, const GeodesicState& deriv, double step) {
  return {addScaled(state.x, deriv.x, step), addScaled(state.u, deriv.u, step)};
}

HORIZON_HD inline GeodesicState rk4Step(const GeodesicState& state, double properStep, double mass) {
  const GeodesicState k1 = geodesicDerivative(state, mass);
  const GeodesicState k2 = geodesicDerivative(addScaled(state, k1, 0.5 * properStep), mass);
  const GeodesicState k3 = geodesicDerivative(addScaled(state, k2, 0.5 * properStep), mass);
  const GeodesicState k4 = geodesicDerivative(addScaled(state, k3, properStep), mass);

  GeodesicState out{};
  out.x = add(state.x, scale(add(add(k1.x, scale(k2.x, 2.0)), add(scale(k3.x, 2.0), k4.x)), properStep / 6.0));
  out.u = add(state.u, scale(add(add(k1.u, scale(k2.u, 2.0)), add(scale(k3.u, 2.0), k4.u)), properStep / 6.0));
  out.x.theta = clamp(out.x.theta, 1.0e-6, kPi - 1.0e-6);
  if (out.x.phi > kTwoPi || out.x.phi < -kTwoPi) {
    out.x.phi = out.x.phi - floor(out.x.phi / kTwoPi) * kTwoPi;
  }
  return out;
}

HORIZON_HD inline double timelikeUtFromNormalization(
    const FourVector& x,
    double ur,
    double utheta,
    double uphi,
    double mass) {
  const double f = metricF(x.r, mass);
  const double s = safeSin(x.theta);
  const double spatial =
      (ur * ur) / f + x.r * x.r * utheta * utheta + x.r * x.r * s * s * uphi * uphi;
  return sqrt((1.0 + spatial) / f);
}

HORIZON_HD inline FourVector circularOrbitFourVelocity(double r, double theta, double mass) {
  const double safeR = r < 3.05 * mass ? 3.05 * mass : r;
  const double denom = sqrt(clamp(1.0 - 3.0 * mass / safeR, 1.0e-10, 1.0e10));
  const double l = sqrt(mass * safeR) / denom;
  const double s = safeSin(theta);
  FourVector u{};
  u.t = 1.0 / denom;
  u.r = 0.0;
  u.theta = 0.0;
  u.phi = l / (safeR * safeR * s * s);
  return u;
}

HORIZON_HD inline void renormalizeTimelike(GeodesicState& state, double mass) {
  state.u.t = timelikeUtFromNormalization(state.x, state.u.r, state.u.theta, state.u.phi, mass);
}

HORIZON_HD inline double adaptiveProperStep(const FourVector& x, const SimulationParams& params) {
  if (!params.enableAdaptiveStep) {
    return params.baseProperTimeStep;
  }

  const double rs = 2.0 * params.geometricMass;
  const double horizonFactor = clamp((x.r - rs) / (8.0 * rs), 0.015, 1.0);
  const double curvature = sqrt(kretschmannScalar(x.r, params.geometricMass));
  const double curvatureFactor = clamp(params.curvatureStepScale / (curvature + 1.0e-12), 0.02, 1.0);
  return clamp(params.baseProperTimeStep * horizonFactor * curvatureFactor,
               params.minProperTimeStep,
               params.maxProperTimeStep);
}

HORIZON_HD inline double gravitationalRedshift(double emitterRadius, double observerRadius, double mass) {
  const double fe = metricF(emitterRadius, mass);
  const double fo = metricF(observerRadius, mass);
  return sqrt(fe / fo);
}

HORIZON_HD inline double relativisticCircularSpeedLocal(double r, double mass) {
  return sqrt(clamp(mass / (r - 2.0 * mass), 0.0, 0.999999));
}

HORIZON_HD inline double lorentzGamma(double beta) {
  return 1.0 / sqrt(clamp(1.0 - beta * beta, 1.0e-12, 1.0));
}

HORIZON_HD inline double dopplerFactor(double beta, double cosEmissionAngle) {
  const double gamma = lorentzGamma(beta);
  return 1.0 / (gamma * (1.0 - beta * cosEmissionAngle));
}

HORIZON_HD inline std::uint32_t pcgHash(std::uint32_t value) {
  std::uint32_t state = value * 747796405u + 2891336453u;
  const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

HORIZON_HD inline double unitRandom(std::uint32_t& state) {
  state = pcgHash(state);
  return static_cast<double>(state) / static_cast<double>(0xffffffffu);
}

HORIZON_HD inline ParticleState makeInitialParticle(
    std::uint32_t index,
    const SimulationParams& params,
    std::uint32_t frameSalt) {
  std::uint32_t seed = pcgHash(index ^ (frameSalt * 0x9e3779b9u));
  const double u0 = unitRandom(seed);
  const double u1 = unitRandom(seed);
  const double u2 = unitRandom(seed);
  const double u3 = unitRandom(seed);
  const double u4 = unitRandom(seed);

  const double inner = params.diskInnerRadius;
  const double outer = params.resetOuterRadius + params.resetWidth * u0;
  const double areaRadius = sqrt(inner * inner + (outer * outer - inner * inner) * u1);
  const double theta = kPi * 0.5 + params.inclinationSpread * (u2 - 0.5);
  const double phi = kTwoPi * u3;

  ParticleState particle{};
  particle.x = {0.0, areaRadius, theta, phi};
  particle.u = circularOrbitFourVelocity(areaRadius, theta, params.geometricMass);
  particle.u.r = params.eccentricitySpread * (u4 - 0.5);
  particle.u.theta = params.inclinationSpread * 0.02 * (unitRandom(seed) - 0.5);
  particle.u.phi *= 1.0 + params.eccentricitySpread * 0.5 * (unitRandom(seed) - 0.5);
  particle.mass = params.particleMass;
  particle.properTime = 0.0;
  particle.flags = ParticleActive;
  particle.seed = seed;
  particle.geodesicDeviation = 1.0e-5;

  GeodesicState normalized{particle.x, particle.u};
  renormalizeTimelike(normalized, params.geometricMass);
  particle.u = normalized.u;

  particle.initialEnergy = conservedEnergy(particle.x, particle.u, params.geometricMass);
  particle.initialAngularMomentum = conservedAngularMomentumZ(particle.x, particle.u);
  particle.initialNorm = fourVelocityNorm(particle.x, particle.u, params.geometricMass);
  particle.energy = particle.initialEnergy;
  particle.angularMomentum = particle.initialAngularMomentum;
  particle.norm = particle.initialNorm;
  particle.dilation = gravitationalDilation(particle.x, params.geometricMass);
  particle.redshift = particle.dilation;
  particle.doppler = 1.0;
  particle.kretschmann = kretschmannScalar(particle.x.r, params.geometricMass);
  particle.tidalStretch = radialTidalEigenvalue(particle.x.r, params.geometricMass);
  return particle;
}

HORIZON_HD inline double vacuumEinsteinResidualAnalytic(const FourVector& x, double mass) {
  (void)x;
  (void)mass;
  return 0.0;
}

}  // namespace horizon::physics
