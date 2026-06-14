#include "horizon/CudaSimulation.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>

#include "horizon/CudaCheck.hpp"

extern "C" void horizonLaunchInitializeParticles(
    horizon::ParticleState* particles,
    horizon::SimulationParams params,
    std::uint32_t frameSalt);

extern "C" void horizonLaunchStepParticles(
    horizon::ParticleState* particles,
    horizon::RenderVertex* renderVertices,
    horizon::Diagnostics* diagnostics,
    horizon::SimulationParams params,
    std::uint32_t frameIndex);

namespace horizon {

CudaSimulation::~CudaSimulation() {
  if (renderResource_ != nullptr) {
    cudaGraphicsUnregisterResource(renderResource_);
  }
  if (deviceParticles_ != nullptr) {
    cudaFree(deviceParticles_);
  }
  if (deviceDiagnostics_ != nullptr) {
    cudaFree(deviceDiagnostics_);
  }
}

void CudaSimulation::initialize(GLuint renderVbo, const SimulationParams& params) {
  if (params.particleCount != kParticleCount) {
    throw std::runtime_error("The Schwarzschild engine is configured for exactly 1,000,000 particles.");
  }

  particleCount_ = params.particleCount;
  registeredVbo_ = renderVbo;

  HORIZON_CUDA_CHECK(cudaMalloc(&deviceParticles_, sizeof(ParticleState) * particleCount_));
  HORIZON_CUDA_CHECK(cudaMalloc(&deviceDiagnostics_, sizeof(Diagnostics)));
  HORIZON_CUDA_CHECK(cudaMemset(deviceDiagnostics_, 0, sizeof(Diagnostics)));
  HORIZON_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
      &renderResource_, registeredVbo_, cudaGraphicsMapFlagsWriteDiscard));

  horizonLaunchInitializeParticles(deviceParticles_, params, 0u);
  HORIZON_CUDA_CHECK(cudaGetLastError());
  HORIZON_CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaSimulation::step(const SimulationParams& params, std::uint32_t frameIndex) {
  if (renderResource_ == nullptr || deviceParticles_ == nullptr) {
    throw std::runtime_error("CudaSimulation::initialize must be called before step.");
  }

  HORIZON_CUDA_CHECK(cudaMemset(deviceDiagnostics_, 0, sizeof(Diagnostics)));
  HORIZON_CUDA_CHECK(cudaGraphicsMapResources(1, &renderResource_, 0));

  auto* renderVertices = static_cast<RenderVertex*>(nullptr);
  std::size_t mappedBytes = 0;
  HORIZON_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&renderVertices), &mappedBytes, renderResource_));

  const std::size_t requiredBytes = sizeof(RenderVertex) * static_cast<std::size_t>(particleCount_);
  if (mappedBytes < requiredBytes) {
    HORIZON_CUDA_CHECK(cudaGraphicsUnmapResources(1, &renderResource_, 0));
    throw std::runtime_error("CUDA-OpenGL VBO is smaller than the particle render buffer.");
  }

  horizonLaunchStepParticles(deviceParticles_, renderVertices, deviceDiagnostics_, params, frameIndex);
  HORIZON_CUDA_CHECK(cudaGetLastError());
  HORIZON_CUDA_CHECK(cudaGraphicsUnmapResources(1, &renderResource_, 0));
}

Diagnostics CudaSimulation::diagnostics() const {
  Diagnostics out{};
  if (deviceDiagnostics_ != nullptr) {
    HORIZON_CUDA_CHECK(cudaMemcpy(&out, deviceDiagnostics_, sizeof(Diagnostics), cudaMemcpyDeviceToHost));
  }
  return out;
}

std::vector<ParticleState> CudaSimulation::downloadParticles(std::size_t maxCount) const {
  const std::size_t count = std::min<std::size_t>(maxCount, particleCount_);
  std::vector<ParticleState> particles(count);
  if (count > 0 && deviceParticles_ != nullptr) {
    HORIZON_CUDA_CHECK(cudaMemcpy(
        particles.data(), deviceParticles_, sizeof(ParticleState) * count, cudaMemcpyDeviceToHost));
  }
  return particles;
}

}  // namespace horizon
