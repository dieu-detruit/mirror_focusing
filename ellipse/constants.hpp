#pragma once

#include <Eigen/Core>

#include <unit/double.hpp>

inline namespace Constants
{

using namespace Unit;

using PhotonFluxDensity = decltype(Amount{} / Area{});
using WaveAmplitude = decltype(std::sqrt(PhotonFluxDensity{}));
inline constexpr PhotonFluxDensity dens_unit{1.0};
inline constexpr WaveAmplitude amp_unit{1.0};

using WaveNumber = decltype(DimensionLessType{} / Length{});

inline constexpr Length lambda = 4.13_nm;
inline constexpr WaveNumber k = 2.0 * M_PI / lambda;

inline constexpr Length a = 2800.060626_mm;  // long axis
inline constexpr Length b = 18.42575822_mm;  // short axis
inline constexpr Length f = 2800.0_mm;       // distance between origin and focal point
inline constexpr Length ML = 120.0_mm;       // Length of mirror
inline constexpr Length WD = 30.0_mm;        // Working distance
inline constexpr Length CD = 74.4_mm;        // distance between camera and the focal plane

inline constexpr Length rotation_center = f / 2.0;

inline constexpr WaveAmplitude source = 1.0 * amp_unit;
inline constexpr Area source_area = 100.0_um2;
inline constexpr Length source_z = -f;

inline constexpr Length focus_length = 200.0_nm;
inline constexpr std::size_t focus_pixel_num = 256;

inline constexpr Length detector_z = f + CD;
inline constexpr Length detector_pixel_size = 15.0_um;
inline constexpr std::size_t detector_pixel_num = 2048;
inline constexpr Length detector_length = detector_pixel_size * detector_pixel_num;

}  // namespace Constants
