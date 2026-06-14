#version 430 core

in vec2 vUv;
out vec4 fragColor;

uniform dvec3 uCameraPosition;
uniform dvec3 uCameraForward;
uniform dvec3 uCameraRight;
uniform dvec3 uCameraUp;
uniform double uMass;
uniform double uDiskInner;
uniform double uDiskOuter;
uniform double uFovY;
uniform double uAspect;
uniform float uTime;
uniform int uOverlayMode;

const double PI = 3.1415926535897932384626433832795;
const double TWO_PI = 6.283185307179586476925286766559;
const int MAX_STEPS = 260;

struct RayState {
  dvec4 x;  // t, r, theta, phi
  dvec4 k;  // dx^mu / d lambda
};

double clampd(double v, double lo, double hi) {
  return min(max(v, lo), hi);
}

double safeSin(double theta) {
  double s = sin(theta);
  return abs(s) < 1.0e-8 ? (s < 0.0 ? -1.0e-8 : 1.0e-8) : s;
}

double metricF(double r) {
  return clampd((r - 2.0 * uMass) / r, 1.0e-12, 1.0e12);
}

dvec3 sphericalToCartesian(dvec4 x) {
  double s = sin(x.z);
  return dvec3(x.y * s * cos(x.w), x.y * cos(x.z), x.y * s * sin(x.w));
}

dvec4 cartesianToSpherical(dvec3 p) {
  double r = max(length(p), 1.0e-9);
  double theta = acos(clampd(p.y / r, -1.0, 1.0));
  double phi = atan(p.z, p.x);
  return dvec4(0.0, r, theta, phi);
}

dvec3 radialBasis(double theta, double phi) {
  return dvec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

dvec3 thetaBasis(double theta, double phi) {
  return dvec3(cos(theta) * cos(phi), -sin(theta), cos(theta) * sin(phi));
}

dvec3 phiBasis(double phi) {
  return dvec3(-sin(phi), 0.0, cos(phi));
}

dvec4 coordinateAcceleration(dvec4 x, dvec4 k) {
  double r = max(x.y, 2.0 * uMass * 1.000001);
  double theta = clampd(x.z, 1.0e-7, PI - 1.0e-7);
  double s = safeSin(theta);
  double c = cos(theta);
  double f = metricF(r);

  double gamma_t_tr = uMass / (r * r * f);
  double gamma_r_tt = uMass * f / (r * r);
  double gamma_r_rr = -uMass / (r * r * f);
  double gamma_r_thth = -r * f;
  double gamma_r_phph = -r * f * s * s;
  double gamma_th_rth = 1.0 / r;
  double gamma_th_phph = -s * c;
  double gamma_ph_rph = 1.0 / r;
  double gamma_ph_thph = c / s;

  return dvec4(
    -2.0 * gamma_t_tr * k.x * k.y,
    -(gamma_r_tt * k.x * k.x + gamma_r_rr * k.y * k.y +
      gamma_r_thth * k.z * k.z + gamma_r_phph * k.w * k.w),
    -(2.0 * gamma_th_rth * k.y * k.z + gamma_th_phph * k.w * k.w),
    -(2.0 * gamma_ph_rph * k.y * k.w + 2.0 * gamma_ph_thph * k.z * k.w)
  );
}

RayState derivative(RayState s) {
  return RayState(s.k, coordinateAcceleration(s.x, s.k));
}

RayState addScaled(RayState a, RayState b, double h) {
  RayState o;
  o.x = a.x + b.x * h;
  o.k = a.k + b.k * h;
  o.x.z = clampd(o.x.z, 1.0e-6, PI - 1.0e-6);
  return o;
}

RayState rk4(RayState s, double h) {
  RayState k1 = derivative(s);
  RayState k2 = derivative(addScaled(s, k1, 0.5 * h));
  RayState k3 = derivative(addScaled(s, k2, 0.5 * h));
  RayState k4 = derivative(addScaled(s, k3, h));
  RayState o;
  o.x = s.x + (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x) * (h / 6.0);
  o.k = s.k + (k1.k + 2.0 * k2.k + 2.0 * k3.k + k4.k) * (h / 6.0);
  o.x.z = clampd(o.x.z, 1.0e-6, PI - 1.0e-6);
  if (o.x.w > TWO_PI || o.x.w < -TWO_PI) {
    o.x.w -= floor(o.x.w / TWO_PI) * TWO_PI;
  }
  return o;
}

RayState initializeRay(dvec3 cameraPosition, dvec3 worldDirection) {
  dvec4 x = cartesianToSpherical(cameraPosition);
  dvec3 er = radialBasis(x.z, x.w);
  dvec3 et = thetaBasis(x.z, x.w);
  dvec3 ep = phiBasis(x.w);
  double f = metricF(x.y);

  double nr = dot(worldDirection, er);
  double nt = dot(worldDirection, et);
  double np = dot(worldDirection, ep);

  dvec4 k;
  k.x = 1.0 / sqrt(f);
  k.y = sqrt(f) * nr;
  k.z = nt / x.y;
  k.w = np / (x.y * safeSin(x.z));
  return RayState(x, k);
}

double kretschmann(double r) {
  return 48.0 * uMass * uMass / pow(r, 6.0);
}

double localCircularBeta(double r) {
  return sqrt(clampd(uMass / max(r - 2.0 * uMass, 1.0e-6), 0.0, 0.999999));
}

double dopplerAtHit(RayState s) {
  double r = s.x.y;
  double theta = s.x.z;
  double f = metricF(r);
  double localR = s.k.y / sqrt(f);
  double localTheta = r * s.k.z;
  double localPhi = r * safeSin(theta) * s.k.w;
  double norm = max(length(dvec3(localR, localTheta, localPhi)), 1.0e-9);
  double cosToObserverAlongOrbit = -localPhi / norm;
  double beta = localCircularBeta(max(r, uDiskInner));
  double gamma = 1.0 / sqrt(max(1.0 - beta * beta, 1.0e-9));
  return 1.0 / (gamma * (1.0 - beta * cosToObserverAlongOrbit));
}

vec3 shiftedDiskColor(double totalShift, double radius, double phi, double delay, double wraps) {
  double rings = 0.58 + 0.42 * sin(radius * 3.7 - delay * 0.22 + 2.0 * sin(phi * 5.0));
  double turbulence = 0.78 + 0.22 * sin(phi * 17.0 + radius * 0.7 + double(uTime) * 0.8);
  double emissivity = pow(uDiskInner / max(radius, uDiskInner), 1.85);
  double boost = pow(clampd(totalShift, 0.05, 5.0), 3.0);

  vec3 red = vec3(1.0, 0.18, 0.045);
  vec3 white = vec3(1.0, 0.78, 0.38);
  vec3 blue = vec3(0.38, 0.62, 1.0);
  float shift = float(clampd(log2(max(totalShift, 1.0e-4)), -2.0, 2.0));
  vec3 color = shift < 0.0 ? mix(white, red, -shift * 0.55) : mix(white, blue, shift * 0.42);
  color *= float(emissivity * boost * rings * turbulence);
  color += vec3(1.0, 0.34, 0.08) * float(max(0.0, wraps - TWO_PI) * 0.018);
  return color;
}

vec3 overlayColor(int mode, double redshift, double r, double wraps, double delay) {
  if (mode == 1) {
    return mix(vec3(0.8, 0.08, 0.02), vec3(0.12, 0.44, 1.0), float(clampd(redshift / 2.5, 0.0, 1.0)));
  }
  if (mode == 2) {
    double f = metricF(r);
    return vec3(float(1.0 - f), float(f), 0.2);
  }
  if (mode == 3) {
    return vec3(float(clampd(log10(1.0 + kretschmann(r) * 1.0e4) / 5.0, 0.0, 1.0)), 0.12, 0.9);
  }
  if (mode == 4) {
    return vec3(float(clampd(wraps / (4.0 * PI), 0.0, 1.0)), 0.25, float(clampd(delay / 120.0, 0.0, 1.0)));
  }
  if (mode == 5) {
    double stable = smoothstep(6.0 * uMass, 6.8 * uMass, r);
    return mix(vec3(1.0, 0.12, 0.02), vec3(0.1, 0.75, 0.35), float(stable));
  }
  return vec3(0.0);
}

void main() {
  dvec2 ndc = dvec2(vUv) * 2.0 - 1.0;
  double tanHalf = tan(uFovY * 0.5);
  dvec3 direction = normalize(
    uCameraForward + uCameraRight * (ndc.x * uAspect * tanHalf) + uCameraUp * (ndc.y * tanHalf));

  RayState ray = initializeRay(uCameraPosition, direction);
  double previousSide = ray.x.z - PI * 0.5;
  double minRadius = ray.x.y;
  double totalPhi = 0.0;
  double previousPhi = ray.x.w;
  double hitRadius = 0.0;
  double hitPhi = 0.0;
  double hitDelay = 0.0;
  double hitShift = 1.0;
  bool hitDisk = false;
  bool captured = false;

  for (int i = 0; i < MAX_STEPS; ++i) {
    double h = clampd(ray.x.y * 0.026, 0.006, 0.82);
    RayState nextRay = rk4(ray, h);
    minRadius = min(minRadius, nextRay.x.y);

    double dphi = abs(nextRay.x.w - previousPhi);
    if (dphi > PI) {
      dphi = TWO_PI - dphi;
    }
    totalPhi += dphi;
    previousPhi = nextRay.x.w;

    if (nextRay.x.y <= 2.0 * uMass * 1.0003) {
      captured = true;
      ray = nextRay;
      break;
    }

    double side = nextRay.x.z - PI * 0.5;
    if (side == 0.0 || side * previousSide < 0.0) {
      double r = nextRay.x.y;
      if (r >= uDiskInner && r <= uDiskOuter) {
        hitDisk = true;
        hitRadius = r;
        hitPhi = nextRay.x.w;
        hitDelay = nextRay.x.x;
        hitShift = sqrt(metricF(r)) * dopplerAtHit(nextRay);
        ray = nextRay;
        break;
      }
    }

    if (nextRay.x.y > uDiskOuter * 1.65 && i > 32 && nextRay.k.y > 0.0) {
      ray = nextRay;
      break;
    }

    previousSide = side;
    ray = nextRay;
  }

  vec3 color = vec3(0.0);
  double photonSphere = 3.0 * uMass;
  double horizon = 2.0 * uMass;

  if (hitDisk) {
    color = shiftedDiskColor(hitShift, hitRadius, hitPhi, hitDelay, totalPhi);
    if (uOverlayMode != 0) {
      color = mix(color, overlayColor(uOverlayMode, hitShift, hitRadius, totalPhi, hitDelay), 0.72);
    }
  } else if (!captured) {
    double star = pow(max(0.0, sin(direction.x * 91.0) * sin(direction.y * 113.0) * sin(direction.z * 71.0)), 48.0);
    color = vec3(star) * vec3(0.34, 0.42, 0.62);
  }

  double photonGlow = exp(-pow((minRadius - photonSphere) / (0.105 * uMass), 2.0));
  color += vec3(1.0, 0.46, 0.09) * float(photonGlow * (0.22 + 0.12 * clampd(totalPhi / TWO_PI, 0.0, 4.0)));

  double shadow = smoothstep(horizon * 1.1, photonSphere * 0.98, minRadius);
  color *= float(1.0 - 0.9 * shadow);
  fragColor = vec4(color, 1.0);
}
