#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define HORIZON_HD __host__ __device__
#else
#define HORIZON_HD
#endif

namespace horizon {

inline constexpr std::uint32_t kParticleCount = 1'000'000u;
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kTwoPi = 2.0 * kPi;

enum ParticleFlags : std::uint32_t {
  ParticleActive = 1u << 0u,
  ParticleCaptured = 1u << 1u,
  ParticleInsideIsco = 1u << 2u,
  ParticleNearPhotonSphere = 1u << 3u,
  ParticleReset = 1u << 4u,
};

struct FourVector {
  double t;
  double r;
  double theta;
  double phi;
};

struct SchwarzschildMetric {
  double g_tt;
  double g_rr;
  double g_thth;
  double g_phph;
};

struct ParticleState {
  FourVector x;                  // Schwarzschild coordinates (t, r, theta, phi).
  FourVector u;                  // Four-velocity dx^mu / d tau.
  double properTime;
  double mass;

  double initialEnergy;
  double initialAngularMomentum;
  double initialNorm;

  double energy;
  double angularMomentum;
  double norm;

  double dilation;
  double redshift;
  double doppler;
  double kretschmann;
  double tidalStretch;
  double geodesicDeviation;

  std::uint32_t flags;
  std::uint32_t seed;
};

struct SimulationParams {
  std::uint32_t particleCount = kParticleCount;
  std::uint32_t resetCapturedParticles = 1u;
  std::uint32_t enableAdaptiveStep = 1u;
  std::uint32_t enableIscoPlunge = 1u;
  std::uint32_t enableDoubleDoubleDiagnostics = 0u;

  double geometricMass = 1.0;     // G M / c^2 in internal length units.
  double baseProperTimeStep = 0.015;
  double maxProperTimeStep = 0.04;
  double minProperTimeStep = 0.00025;
  double curvatureStepScale = 0.025;
  double plungeDamping = 0.35;
  double diskInnerRadius = 6.15;
  double diskOuterRadius = 80.0;
  double resetOuterRadius = 78.0;
  double resetWidth = 12.0;
  double inclinationSpread = 0.035;
  double eccentricitySpread = 0.045;
  double particleMass = 1.0;
  double physicalBlackHoleMassSolar = 4.1e6;
  double coordinateTime = 0.0;
};

struct Diagnostics {
  double sumEnergyDrift = 0.0;
  double maxEnergyDrift = 0.0;
  double sumAngularMomentumDrift = 0.0;
  double maxAngularMomentumDrift = 0.0;
  double sumNormDrift = 0.0;
  double maxNormDrift = 0.0;
  double sumDilation = 0.0;
  double sumRedshift = 0.0;
  double maxKretschmann = 0.0;
  double maxTidalStretch = 0.0;
  double vacuumEinsteinResidual = 0.0;
  std::uint32_t activeCount = 0u;
  std::uint32_t capturedCount = 0u;
  std::uint32_t insideIscoCount = 0u;
  std::uint32_t nearPhotonSphereCount = 0u;
};

struct RenderVertex {
  float x;
  float y;
  float z;
  float size;
  float redshift;
  float dilation;
  float curvature;
  float flags;
};

HORIZON_HD inline double schwarzschildRadius(const SimulationParams& params) {
  return 2.0 * params.geometricMass;
}

HORIZON_HD inline double iscoRadius(const SimulationParams& params) {
  return 6.0 * params.geometricMass;
}

HORIZON_HD inline double photonSphereRadius(const SimulationParams& params) {
  return 3.0 * params.geometricMass;
}

}  // namespace horizon
