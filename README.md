# Horizon
`horizon` is a CUDA/OpenGL Schwarzschild black hole simulation and visualization project. It advances exactly 1,000,000 independent timelike particle geodesics on the GPU and renders them through CUDA-OpenGL interop while a fullscreen GLSL pass ray-traces null Schwarzschild geodesics for gravitational lensing, photon-sphere glow, redshift, Doppler boosting, and Shapiro-delay modulation.

## Physics Model

- Internal units are geometric units with `G = c = M = 1`; physical mass scaling is kept in `SimulationParams`.
- Particle states are four-vectors `(t, r, theta, phi)` plus four-velocity `dx^mu / d tau`.
- The Schwarzschild metric tensor is explicit and analytic Christoffel-symbol contractions drive the geodesic equation.
- The CUDA kernel uses fourth-order Runge-Kutta integration in proper time with adaptive step reduction near high curvature and the event horizon.
- Conserved specific energy, angular momentum, four-velocity norm, proper time, redshift, tidal stretch, geodesic-deviation proxy, and Kretschmann scalar are tracked per particle.
- The ISCO is `r = 6GM/c^2`; inside it, a configurable plunge perturbation models accretion stress and rapidly sends particles toward the event horizon at `r = 2GM/c^2`.
- The photon sphere is `r = 3GM/c^2`; the lensing shader naturally creates unstable near-circular photon paths by integrating null geodesics.

## Build

Required dependencies:

- CMake 3.24+
- CUDA Toolkit with OpenGL interop support
- OpenGL 4.3+
- GLFW 3
- GLEW

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Optional double-double horizon diagnostics:

```bash
cmake -S . -B build-dd -DCMAKE_BUILD_TYPE=Release -DHORIZON_ENABLE_DOUBLE_DOUBLE=ON
cmake --build build-dd -j
```

## Run

```bash
./build/horizon
```

Controls:

- Left drag: rotate arcball camera
- Right or middle drag: pan
- Scroll: zoom deeply toward the accretion disk and horizon
- Space: pause or resume simulation
- `P`: toggle particle layer
- `E`: export a 20,000-particle CSV snapshot
- `0`: beauty render
- `1`: redshift map
- `2`: metric coefficient overlay
- `3`: curvature/Kretschmann overlay
- `4`: photon wrap and Shapiro-delay overlay
- `5`: orbital stability/ISCO overlay

Diagnostics are appended to `exports/diagnostics.csv`; snapshots are written to `exports/particles_<frame>.csv`.

## Source Map

- `src/physics/Relativity.cuh`: metric tensor, Christoffel contractions, geodesic RK4, circular orbit, redshift, tidal, curvature, and validation helpers.
- `src/cuda/SchwarzschildKernels.cu`: one-thread-per-particle CUDA initialization and integration kernels plus diagnostics reduction.
- `src/CudaSimulation.cpp`: host-side CUDA memory and `cudaGraphicsGLRegisterBuffer` interop management.
- `src/main.cpp`: GLFW/OpenGL application, camera controls, rendering loop, and export triggers.
- `shaders/lensing.frag`: per-pixel null-geodesic ray integration for lensing, redshift, Doppler boosting, photon sphere, and Shapiro-delay visualization.
- `tests/validation.cu`: automated checks against circular timelike orbits, photon sphere null geodesics, redshift, curvature, vacuum Schwarzschild residual, and periapsis precession.

## Notes

The particle engine is double precision on CUDA. Rendering buffers are floats because OpenGL point rasterization consumes screen-space positions; the simulation remains in compact geometric coordinates near the black hole to avoid astronomical-scale precision loss. The fullscreen lensing shader uses GLSL `double` geodesic integration, which favors physical stability over maximum fragment throughput.
