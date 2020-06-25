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

using WaveNumber = decltype(DimensionlessType{} / Length{});

inline constexpr Length lambda = 4.13_nm;
inline constexpr WaveNumber k = 2.0 * M_PI / lambda;

inline constexpr Length ML = 1.0_mm;

inline constexpr WaveAmplitude source = 1.0 * amp_unit;
inline constexpr Area source_area = 100.0_um2;
inline constexpr Length source_z = -100.0_mm;
inline constexpr Length source_y = 10.0_mm;

inline constexpr Length reflection_length = 50.0_mm;
inline constexpr Length reflection_z = 100.0_mm;
inline constexpr std::size_t reflection_pixel_num = 256;


}  // namespace Constants
