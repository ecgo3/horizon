#include "horizon/CsvExporter.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace horizon {

CsvExporter::CsvExporter(std::filesystem::path directory)
    : directory_(std::move(directory)), diagnosticsPath_(directory_ / "diagnostics.csv") {
  std::filesystem::create_directories(directory_);
  if (!std::filesystem::exists(diagnosticsPath_)) {
    std::ofstream out(diagnosticsPath_);
    out << "frame,wall_time_seconds,active,captured,inside_isco,near_photon_sphere,"
           "mean_energy_drift,max_energy_drift,mean_angular_momentum_drift,"
           "max_angular_momentum_drift,mean_norm_drift,max_norm_drift,mean_dilation,"
           "mean_redshift,max_kretschmann,max_tidal_stretch,vacuum_einstein_residual\n";
  }
}

void CsvExporter::appendDiagnostics(std::uint64_t frame, double wallTimeSeconds, const Diagnostics& d) {
  std::ofstream out(diagnosticsPath_, std::ios::app);
  if (!out) {
    throw std::runtime_error("Unable to append diagnostics CSV: " + diagnosticsPath_.string());
  }

  const double active = static_cast<double>(d.activeCount == 0u ? 1u : d.activeCount);
  out << frame << ',' << std::setprecision(10) << wallTimeSeconds << ',' << d.activeCount << ','
      << d.capturedCount << ',' << d.insideIscoCount << ',' << d.nearPhotonSphereCount << ','
      << d.sumEnergyDrift / active << ',' << d.maxEnergyDrift << ','
      << d.sumAngularMomentumDrift / active << ',' << d.maxAngularMomentumDrift << ','
      << d.sumNormDrift / active << ',' << d.maxNormDrift << ',' << d.sumDilation / active << ','
      << d.sumRedshift / active << ',' << d.maxKretschmann << ',' << d.maxTidalStretch << ','
      << d.vacuumEinsteinResidual << '\n';
}

void CsvExporter::writeParticleSnapshot(std::uint64_t frame, std::span<const ParticleState> particles) const {
  const std::filesystem::path path = directory_ / ("particles_" + std::to_string(frame) + ".csv");
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Unable to write particle CSV: " + path.string());
  }

  out << "index,t,r,theta,phi,ut,ur,utheta,uphi,proper_time,energy,angular_momentum,norm,"
         "dilation,redshift,doppler,kretschmann,tidal_stretch,geodesic_deviation,flags\n";
  out << std::setprecision(17);
  for (std::size_t i = 0; i < particles.size(); ++i) {
    const ParticleState& p = particles[i];
    out << i << ',' << p.x.t << ',' << p.x.r << ',' << p.x.theta << ',' << p.x.phi << ','
        << p.u.t << ',' << p.u.r << ',' << p.u.theta << ',' << p.u.phi << ',' << p.properTime
        << ',' << p.energy << ',' << p.angularMomentum << ',' << p.norm << ',' << p.dilation
        << ',' << p.redshift << ',' << p.doppler << ',' << p.kretschmann << ',' << p.tidalStretch
        << ',' << p.geodesicDeviation << ',' << p.flags << '\n';
  }
}

}  // namespace horizon
