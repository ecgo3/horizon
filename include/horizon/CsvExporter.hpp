#pragma once

#include <filesystem>
#include <span>

#include "horizon/SimulationTypes.hpp"

namespace horizon {

class CsvExporter {
 public:
  explicit CsvExporter(std::filesystem::path directory);

  void appendDiagnostics(std::uint64_t frame, double wallTimeSeconds, const Diagnostics& diagnostics);
  void writeParticleSnapshot(std::uint64_t frame, std::span<const ParticleState> particles) const;

 private:
  std::filesystem::path directory_;
  std::filesystem::path diagnosticsPath_;
};

}  // namespace horizon
