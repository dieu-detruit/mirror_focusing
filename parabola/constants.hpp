#pragma once

#include <Eigen/Core>

#include <unit/double.hpp>

inline namespace Constants
{

using namespace Unit;

using PhotonFluxDensity = Amount;
using WaveAmplitude = decltype(std::sqrt(PhotonFluxDensity{}));
inline constexpr PhotonFluxDensity dens_unit{1.0};
inline constexpr WaveAmplitude amp_unit{1.0};

using WaveNumber = decltype(DimensionLessType{} / Length{});

Length lambda = 10.0_nm;
WaveNumber k = 2.0 * M_PI / lambda;

inline constexpr WaveAmplitude source = 1.0 * amp_unit;
inline constexpr Area source_area = 100.0_um2;
inline constexpr Length source_d = 300.0_mm;
inline constexpr Length source_z = 0.0_mm;

inline constexpr Length mirror_offset_z = 100.0_mm;

inline constexpr Length mirror_focus_z = 0.0_mm;
inline constexpr Length p = mirror_offset_z - mirror_focus_z;

inline constexpr Length d_min = 40.0_mm;
inline constexpr Area d_coef = 0.05_mm * d_min;
inline constexpr std::size_t d_division = 100;
inline constexpr std::size_t theta_division = 100;
inline constexpr Angle theta_step = 2 * M_PI / theta_division * 1.0_rad;

inline constexpr Length reflection_length = 1000.0_mm;
inline constexpr std::size_t reflection_num = 256;


}  // namespace Constants
