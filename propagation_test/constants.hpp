#pragma once

#include <unit/double.hpp>

inline namespace Constants
{

using namespace Unit;

using PhotonFluxDensity = Amount;
using WaveAmplitude = decltype(std::sqrt(PhotonFluxDensity{}));
inline constexpr PhotonFluxDensity dens_unit{1.0};
inline constexpr WaveAmplitude amp_unit{1.0};

using WaveNumber = decltype(DimensionLessType{} / Length{});

Length lambda = 700.0_nm;
WaveNumber k = 2.0 * M_PI / lambda;

inline constexpr Length diffraction_length = 10.0_mm;
inline constexpr std::size_t diffraction_pixel_num = 128;
inline constexpr Length diffraction_pixel_size = diffraction_length / diffraction_pixel_num;
inline constexpr Area diffraction_pixel_area = diffraction_pixel_size * diffraction_pixel_size;

inline constexpr WaveAmplitude aperture_wave = 1.0 * amp_unit;
inline constexpr Length aperture_length = 1.0_mm;
inline constexpr std::size_t aperture_pixel_num = 64;


}  // namespace Constants
