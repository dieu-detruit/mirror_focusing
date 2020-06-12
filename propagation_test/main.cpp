#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>

#include <fftw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>
#include <unit/double.hpp>

#include <string>
#include <unit/double.hpp>

#include "constants.hpp"

int main()
{
    Grid::DynamicRange<Length> aperture_range{-0.5 * aperture_length, 0.5 * aperture_length, aperture_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> aperture{aperture_range, aperture_range};

    Grid::DynamicRange<Length> diffraction_range{-0.5 * diffraction_length, 0.5 * diffraction_length, diffraction_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> diffraction{diffraction_range, diffraction_range};

    aperture.fill(aperture_wave);

    //{
    //std::ofstream file("diffraction0.txt");

    //for (auto& x : diffraction.line(0)) {
    //for (auto& y : diffraction.line(1)) {
    //file << x / 1.0_m << ' '
    //<< y / 1.0_m << ' '
    //<< std::arg(aperture.at(x, y)) / 1.0_rad << ' '
    //<< std::abs(aperture.at(x, y)) / amp_unit << std::endl;
    //}
    //file << std::endl;
    //}
    //}

    for (auto& z : std::array<Length, 3>{{300.0_mm, 500.0_mm, 1.0_m}}) {

        std::cout << z << std::endl;

        std::ofstream file("diffraction" + std::to_string(int(z / 1.0_mm)) + ".txt");

        diffraction.fill(0.0 * amp_unit);
        for (auto [x, y, d] : Grid::zip(diffraction.lines(), diffraction)) {
            for (auto [xi, eta, a] : Grid::zip(aperture.lines(), aperture)) {
                Length r = std::hypot(x - xi, y - eta, z);
                d += z / r / r * aperture_wave * std::exp(-1.0i * k * r) / 1.0i / lambda * diffraction_pixel_area;
            }
        }

        for (auto& x : diffraction.line(0)) {
            for (auto& y : diffraction.line(1)) {
                file << x / 1.0_m << ' '
                     << y / 1.0_m << ' '
                     << std::arg(diffraction.at(x, y)) / 1.0_rad << ' '
                     << std::abs(diffraction.at(x, y)) / amp_unit << std::endl;
            }
            file << std::endl;
        }
    }

    return 0;
}
