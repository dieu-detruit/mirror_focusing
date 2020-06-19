#pragma once

#include <cmath>

#include <Eigen/Core>

#include <unit/double.hpp>

inline namespace Constants
{

using namespace Unit;

template <class T1, class T2, class T3>
Eigen::Vector3d vectorize(T1 x, T2 y, T3 z)
{
    double _x, _y, _z;
    if constexpr (std::is_same_v<T1, Length>) {
        _x = x / 1.0_m;
    } else {
        _x = x;
    }
    if constexpr (std::is_same_v<T2, Length>) {
        _y = y / 1.0_m;
    } else {
        _y = y;
    }
    if constexpr (std::is_same_v<T3, Length>) {
        _z = z / 1.0_m;
    } else {
        _z = z;
    }

    return Eigen::Vector3d{_x, _y, _z};
}

constexpr Length f = 20.0_mm;
constexpr Length L1 = 18.0_m;
constexpr Length L2 = 70.0_m;
constexpr Length ML = 60.0_mm;
constexpr Length f1 = 62.9_m;
constexpr Length f2 = 7.0_m;
constexpr Length f3 = 17.98_m;
constexpr Length f4 = 0.019_m;
constexpr Angle phi = 0.17_mrad;
constexpr Angle theta1 = 250.0_mrad;
constexpr Angle theta2 = 100.0_mrad;

const double sin_phi = std::sin(phi);
const double cos_phi = std::cos(phi);

const Length a = L1 / 2.0
                 * std::sqrt(
                     (1.0 + std::sqrt(1.0 - std::pow((1.0 - 2.0 * f / L1) * std::sin(2.0 * theta1), 2)))
                     / (1.0 + std::cos(2.0 * theta1)));
const Length b = std::sqrt(a * a - L1 * L1 / 4.0);
const Length f_dash
    = f * cos_phi
      - b * sin_phi * std::sqrt(1.0 - (L1 / 2.0 - f).pow<2>() / a / a);

const auto A0 = a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi;
const auto B0 = -L1 * b * b * sin_phi / 2.0;
const auto B1 = (a * a - b * b) * sin_phi * cos_phi;
const auto C0 = b * b * (L1 * L1 / 4.0 - a * a);
const auto C1 = b * b * L1 * cos_phi;
const auto C2 = b * b * cos_phi * cos_phi + a * a * sin_phi * sin_phi;
const auto z = [](Length x) {
    return (-B1 * x
               - B0
               + std::sqrt(
                   (B1 * B1 - A0 * C2) * x * x + (2.0 * B0 * B1 - A0 * C1) * x - A0 * C0))
           / A0;
};

const Length x_dash = L1 * (f_dash * sin_phi - z(-f_dash) * cos_phi) / (z(-f_dash) - L1 * sin_phi);
const Angle theta_dash = std::atan2(z(-f_dash), -f_dash - x_dash);

const Length x_r = -L1 * cos_phi;
const Eigen::Vector3d pos_s
    = (double)((L2 - std::hypot(-L1 * cos_phi - x_dash, L1* sin_phi)) / 1.0_m)
          * vectorize(-std::cos(2.0 * theta2 - theta_dash), 0.0, std::sin(2.0 * theta2 - theta_dash))
      + vectorize(x_dash, 0.0, 0.0);
const Length x_s = pos_s(0) * 1.0_m;
const Length z_s = pos_s(2) * 1.0_m;
const Length d = L1 * sin_phi;
const auto p = [](Angle alpha, Angle beta) {
    double sin_beta = std::sin(beta);
    double cos_alpha = std::cos(alpha);
    return (d * d + (x_s - x_r).pow<2>() + z_s * z_s - 2.0 * d * z_s * sin_beta - L2 * L2)
           / 2.0 / (L2 + (x_s - x_r) * std::sin(alpha) + z_s * sin_beta * cos_alpha - d * cos_alpha);
};

}  // namespace Constants
