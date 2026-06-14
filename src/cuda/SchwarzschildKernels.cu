#include <cuda_runtime.h>

#include <cstdint>

#include "horizon/SimulationTypes.hpp"
#include "physics/Relativity.cuh"

namespace {

constexpr int kBlockSize = 256;

__device__ double atomicMaxDouble(double* address, double value) {
  auto* addressAsUll = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *addressAsUll;
  unsigned long long int assumed = 0;

  do {
    assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    if (current >= value) {
      break;
    }
    old = atomicCAS(addressAsUll, assumed, __double_as_longlong(value));
  } while (assumed != old);

  return __longlong_as_double(static_cast<long long>(old));
}

__device__ void accumulateBlockDiagnostics(
    horizon::Diagnostics* diagnostics,
    double energyDrift,
    double maxEnergyDrift,
    double angularMomentumDrift,
    double maxAngularMomentumDrift,
    double normDrift,
    double maxNormDrift,
    double dilation,
    double redshift,
    double maxKretschmann,
    double maxTidalStretch,
    unsigned int active,
    unsigned int captured,
    unsigned int insideIsco,
    unsigned int nearPhotonSphere) {
  atomicAdd(&diagnostics->sumEnergyDrift, energyDrift);
  atomicMaxDouble(&diagnostics->maxEnergyDrift, maxEnergyDrift);
  atomicAdd(&diagnostics->sumAngularMomentumDrift, angularMomentumDrift);
  atomicMaxDouble(&diagnostics->maxAngularMomentumDrift, maxAngularMomentumDrift);
  atomicAdd(&diagnostics->sumNormDrift, normDrift);
  atomicMaxDouble(&diagnostics->maxNormDrift, maxNormDrift);
  atomicAdd(&diagnostics->sumDilation, dilation);
  atomicAdd(&diagnostics->sumRedshift, redshift);
  atomicMaxDouble(&diagnostics->maxKretschmann, maxKretschmann);
  atomicMaxDouble(&diagnostics->maxTidalStretch, maxTidalStretch);
  atomicAdd(&diagnostics->activeCount, active);
  atomicAdd(&diagnostics->capturedCount, captured);
  atomicAdd(&diagnostics->insideIscoCount, insideIsco);
  atomicAdd(&diagnostics->nearPhotonSphereCount, nearPhotonSphere);
}

__global__ void initializeParticlesKernel(
    horizon::ParticleState* particles,
    horizon::SimulationParams params,
    std::uint32_t frameSalt) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= params.particleCount) {
    return;
  }
  particles[index] = horizon::physics::makeInitialParticle(index, params, frameSalt);
}

__global__ void stepParticlesKernel(
    horizon::ParticleState* particles,
    horizon::RenderVertex* renderVertices,
    horizon::Diagnostics* diagnostics,
    horizon::SimulationParams params,
    std::uint32_t frameIndex) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ double sumEnergy[kBlockSize];
  __shared__ double maxEnergy[kBlockSize];
  __shared__ double sumAngular[kBlockSize];
  __shared__ double maxAngular[kBlockSize];
  __shared__ double sumNorm[kBlockSize];
  __shared__ double maxNorm[kBlockSize];
  __shared__ double sumDilation[kBlockSize];
  __shared__ double sumRedshift[kBlockSize];
  __shared__ double maxK[kBlockSize];
  __shared__ double maxTidal[kBlockSize];
  __shared__ unsigned int active[kBlockSize];
  __shared__ unsigned int captured[kBlockSize];
  __shared__ unsigned int insideIsco[kBlockSize];
  __shared__ unsigned int nearPhotonSphere[kBlockSize];

  double localEnergy = 0.0;
  double localMaxEnergy = 0.0;
  double localAngular = 0.0;
  double localMaxAngular = 0.0;
  double localNorm = 0.0;
  double localMaxNorm = 0.0;
  double localDilation = 0.0;
  double localRedshift = 0.0;
  double localMaxK = 0.0;
  double localMaxTidal = 0.0;
  unsigned int localActive = 0u;
  unsigned int localCaptured = 0u;
  unsigned int localInsideIsco = 0u;
  unsigned int localNearPhotonSphere = 0u;

  if (index < params.particleCount) {
    horizon::ParticleState particle = particles[index];
    const double mass = params.geometricMass;
    const double rs = 2.0 * mass;
    const double isco = 6.0 * mass;
    const double photonSphere = 3.0 * mass;

    if ((particle.flags & horizon::ParticleCaptured) != 0u && params.resetCapturedParticles != 0u) {
      particle = horizon::physics::makeInitialParticle(index, params, frameIndex + particle.seed);
      particle.flags |= horizon::ParticleReset;
    }

    if ((particle.flags & horizon::ParticleCaptured) == 0u) {
      double properStep = horizon::physics::adaptiveProperStep(particle.x, params);
      particle.flags = horizon::ParticleActive;

      if (particle.x.r <= isco) {
        particle.flags |= horizon::ParticleInsideIsco;
        localInsideIsco = 1u;

        if (params.enableIscoPlunge != 0u) {
          const double plungeSpeed = -0.22 * sqrt(mass / horizon::physics::clamp(particle.x.r, rs * 1.001, 1.0e9));
          if (particle.u.r > plungeSpeed) {
            particle.u.r = 0.65 * particle.u.r + 0.35 * plungeSpeed;
          }
          particle.u.phi *= exp(-params.plungeDamping * properStep);
          horizon::physics::GeodesicState plungeState{particle.x, particle.u};
          horizon::physics::renormalizeTimelike(plungeState, mass);
          particle.u = plungeState.u;
        }
      }

      horizon::physics::GeodesicState state{particle.x, particle.u};
      state = horizon::physics::rk4Step(state, properStep, mass);
      particle.x = state.x;
      particle.u = state.u;
      particle.properTime += properStep;

      if (particle.x.r <= rs * 1.0000001) {
        particle.flags = horizon::ParticleCaptured;
        localCaptured = 1u;
        if (params.resetCapturedParticles != 0u) {
          particle = horizon::physics::makeInitialParticle(index, params, frameIndex + 0xa511e9b3u);
          particle.flags |= horizon::ParticleReset;
        }
      }

      particle.energy = horizon::physics::conservedEnergy(particle.x, particle.u, mass);
      particle.angularMomentum = horizon::physics::conservedAngularMomentumZ(particle.x, particle.u);
      particle.norm = horizon::physics::fourVelocityNorm(particle.x, particle.u, mass);
      particle.dilation = horizon::physics::gravitationalDilation(particle.x, mass);
      particle.kretschmann = horizon::physics::kretschmannScalar(particle.x.r, mass);

      const double beta = horizon::physics::relativisticCircularSpeedLocal(
          horizon::physics::clamp(particle.x.r, isco + 1.0e-6, params.diskOuterRadius), mass);
      const double observerCosine = -sin(particle.x.phi);
      particle.doppler = horizon::physics::dopplerFactor(beta, observerCosine);
      particle.redshift = particle.dilation * particle.doppler;
      particle.tidalStretch = horizon::physics::radialTidalEigenvalue(particle.x.r, mass);
      particle.geodesicDeviation *= exp(horizon::physics::clamp(particle.tidalStretch * properStep, -0.02, 0.02));

      if (fabs(particle.x.r - photonSphere) < 0.08 * mass) {
        particle.flags |= horizon::ParticleNearPhotonSphere;
        localNearPhotonSphere = 1u;
      }

      if ((particle.flags & horizon::ParticleCaptured) == 0u) {
        localActive = 1u;
      }

      const double energyDenom = fabs(particle.initialEnergy) + 1.0e-18;
      const double angularDenom = fabs(particle.initialAngularMomentum) + 1.0e-18;
      localEnergy = fabs((particle.energy - particle.initialEnergy) / energyDenom);
      localMaxEnergy = localEnergy;
      localAngular = fabs((particle.angularMomentum - particle.initialAngularMomentum) / angularDenom);
      localMaxAngular = localAngular;
      localNorm = fabs(particle.norm - particle.initialNorm);
      localMaxNorm = localNorm;
      localDilation = particle.dilation;
      localRedshift = particle.redshift;
      localMaxK = particle.kretschmann;
      localMaxTidal = fabs(particle.tidalStretch * particle.geodesicDeviation);

      const horizon::physics::Cartesian cart = horizon::physics::sphericalToCartesian(particle.x);
      const float visible = (particle.flags & horizon::ParticleCaptured) == 0u ? 1.0f : 0.0f;
      renderVertices[index] = {
          static_cast<float>(cart.x),
          static_cast<float>(cart.y),
          static_cast<float>(cart.z),
          visible * static_cast<float>(horizon::physics::clamp(2.5 / sqrt(particle.x.r), 0.5, 4.5)),
          static_cast<float>(horizon::physics::clamp(particle.redshift, 0.1, 4.0)),
          static_cast<float>(horizon::physics::clamp(particle.dilation, 0.0, 1.0)),
          static_cast<float>(horizon::physics::clamp(log10(1.0 + particle.kretschmann * 1.0e4), 0.0, 6.0)),
          static_cast<float>(particle.flags)};
    } else {
      renderVertices[index] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                               static_cast<float>(particle.flags)};
    }

    particles[index] = particle;
  }

  const int lane = threadIdx.x;
  sumEnergy[lane] = localEnergy;
  maxEnergy[lane] = localMaxEnergy;
  sumAngular[lane] = localAngular;
  maxAngular[lane] = localMaxAngular;
  sumNorm[lane] = localNorm;
  maxNorm[lane] = localMaxNorm;
  sumDilation[lane] = localDilation;
  sumRedshift[lane] = localRedshift;
  maxK[lane] = localMaxK;
  maxTidal[lane] = localMaxTidal;
  active[lane] = localActive;
  captured[lane] = localCaptured;
  insideIsco[lane] = localInsideIsco;
  nearPhotonSphere[lane] = localNearPhotonSphere;
  __syncthreads();

  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      sumEnergy[lane] += sumEnergy[lane + stride];
      maxEnergy[lane] = fmax(maxEnergy[lane], maxEnergy[lane + stride]);
      sumAngular[lane] += sumAngular[lane + stride];
      maxAngular[lane] = fmax(maxAngular[lane], maxAngular[lane + stride]);
      sumNorm[lane] += sumNorm[lane + stride];
      maxNorm[lane] = fmax(maxNorm[lane], maxNorm[lane + stride]);
      sumDilation[lane] += sumDilation[lane + stride];
      sumRedshift[lane] += sumRedshift[lane + stride];
      maxK[lane] = fmax(maxK[lane], maxK[lane + stride]);
      maxTidal[lane] = fmax(maxTidal[lane], maxTidal[lane + stride]);
      active[lane] += active[lane + stride];
      captured[lane] += captured[lane + stride];
      insideIsco[lane] += insideIsco[lane + stride];
      nearPhotonSphere[lane] += nearPhotonSphere[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    accumulateBlockDiagnostics(
        diagnostics,
        sumEnergy[0],
        maxEnergy[0],
        sumAngular[0],
        maxAngular[0],
        sumNorm[0],
        maxNorm[0],
        sumDilation[0],
        sumRedshift[0],
        maxK[0],
        maxTidal[0],
        active[0],
        captured[0],
        insideIsco[0],
        nearPhotonSphere[0]);
  }
}

}  // namespace

extern "C" void horizonLaunchInitializeParticles(
    horizon::ParticleState* particles,
    horizon::SimulationParams params,
    std::uint32_t frameSalt) {
  const int blocks = static_cast<int>((params.particleCount + kBlockSize - 1u) / kBlockSize);
  initializeParticlesKernel<<<blocks, kBlockSize>>>(particles, params, frameSalt);
}

extern "C" void horizonLaunchStepParticles(
    horizon::ParticleState* particles,
    horizon::RenderVertex* renderVertices,
    horizon::Diagnostics* diagnostics,
    horizon::SimulationParams params,
    std::uint32_t frameIndex) {
  const int blocks = static_cast<int>((params.particleCount + kBlockSize - 1u) / kBlockSize);
  stepParticlesKernel<<<blocks, kBlockSize>>>(particles, renderVertices, diagnostics, params, frameIndex);
}
