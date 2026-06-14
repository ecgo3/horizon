#pragma once

#include <GL/glew.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "horizon/SimulationTypes.hpp"

struct cudaGraphicsResource;

namespace horizon {

class CudaSimulation {
 public:
  CudaSimulation() = default;
  ~CudaSimulation();

  CudaSimulation(const CudaSimulation&) = delete;
  CudaSimulation& operator=(const CudaSimulation&) = delete;

  CudaSimulation(CudaSimulation&&) = delete;
  CudaSimulation& operator=(CudaSimulation&&) = delete;

  void initialize(GLuint renderVbo, const SimulationParams& params);
  void step(const SimulationParams& params, std::uint32_t frameIndex);
  [[nodiscard]] Diagnostics diagnostics() const;
  [[nodiscard]] std::vector<ParticleState> downloadParticles(std::size_t maxCount) const;

 private:
  ParticleState* deviceParticles_ = nullptr;
  Diagnostics* deviceDiagnostics_ = nullptr;
  cudaGraphicsResource* renderResource_ = nullptr;
  GLuint registeredVbo_ = 0;
  std::uint32_t particleCount_ = 0;
};

}  // namespace horizon
