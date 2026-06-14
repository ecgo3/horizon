#pragma once

#include <cmath>

#include "horizon/SimulationTypes.hpp"

namespace horizon::physics {

struct DoubleDouble {
  double hi;
  double lo;
};

HORIZON_HD inline DoubleDouble twoSum(double a, double b) {
  const double s = a + b;
  const double bb = s - a;
  const double err = (a - (s - bb)) + (b - bb);
  return {s, err};
}

HORIZON_HD inline DoubleDouble quickTwoSum(double a, double b) {
  const double s = a + b;
  return {s, b - (s - a)};
}

HORIZON_HD inline DoubleDouble twoProd(double a, double b) {
  const double p = a * b;
#if defined(__CUDA_ARCH__)
  const double err = fma(a, b, -p);
#else
  const double err = std::fma(a, b, -p);
#endif
  return {p, err};
}

HORIZON_HD inline DoubleDouble add(DoubleDouble a, DoubleDouble b) {
  const DoubleDouble s = twoSum(a.hi, b.hi);
  const DoubleDouble t = twoSum(a.lo, b.lo);
  const DoubleDouble u = quickTwoSum(s.hi, s.lo + t.hi);
  return quickTwoSum(u.hi, u.lo + t.lo);
}

HORIZON_HD inline DoubleDouble sub(DoubleDouble a, DoubleDouble b) {
  return add(a, {-b.hi, -b.lo});
}

HORIZON_HD inline DoubleDouble mul(DoubleDouble a, DoubleDouble b) {
  const DoubleDouble p = twoProd(a.hi, b.hi);
  return quickTwoSum(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}

HORIZON_HD inline DoubleDouble fromDouble(double value) {
  return {value, 0.0};
}

HORIZON_HD inline double toDouble(DoubleDouble value) {
  return value.hi + value.lo;
}

HORIZON_HD inline double horizonSeparation(double r, double radius) {
#if defined(HORIZON_ENABLE_DOUBLE_DOUBLE)
  return toDouble(sub(fromDouble(r), fromDouble(radius)));
#else
  return r - radius;
#endif
}

}  // namespace horizon::physics
