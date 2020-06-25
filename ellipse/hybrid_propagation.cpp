#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fftw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <grid/algorithm.hpp>
#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>

#include <unit/double.hpp>
#include <unit/impl/std_overload.hpp>

#include "constants.hpp"

std::string datadir = "data_hybrid_propagation";

template <class complex_grid_vector>
void print_field(complex_grid_vector& field, std::string filename)
{
    std::ofstream file(datadir + "/" + filename);
    for (auto& x : field.line(0)) {
        for (auto& y : field.line(1)) {
            file << x.value << ' ' << y.value << ' '
                 << std::arg(field.at(x, y)).value << ' '
                 << std::norm(field.at(x, y)).value << std::endl;
        }
        file << std::endl;
    }
}

Eigen::Vector3d vectorize(Length x, Length y, Length z)
{
    return Eigen::Vector3d{x.value, y.value, z.value};
}
Eigen::Vector2d vectorize(Length x, Length y)
{
    return Eigen::Vector2d{x.value, y.value};
}

int main()
{
    if (!fftw_init_threads()) {
        std::cerr << "error at initialization of threads" << std::endl;
        return 1;
    }
    fftw_plan_with_nthreads(4);

    Grid::parallelize();

    std::filesystem::create_directory(datadir);

    constexpr Length focus_length = 200.0_nm;
    constexpr Length exit_length = focus_pixel_num * lambda * WD / focus_length;

    Grid::DynamicRange<Length> exit_range{-0.5 * exit_length, 0.5 * exit_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> exit{exit_range, exit_range};

    Grid::GridVector<Eigen::Vector3d, Length, 2> reflect_points{exit_range, exit_range};

    Eigen::Vector3d source_pos = vectorize(0.0_m, 0.0_m, -f);
    Eigen::Vector3d focus_pos = vectorize(0.0_m, 0.0_m, f);

    constexpr double coef_0 = 1.0 - f * f / (a * a);
    constexpr double coef_1 = f * WD / (a * a);
    constexpr double coef_1_sq = coef_1 * coef_1;

    std::cout << exit_radius << std::endl;
    std::cout << exit_length << std::endl;

    for (auto [x, y] : exit.lines()) {
        if (y > 0.0_m or y < x) {  // 対称性
            if (std::abs(x) < std::abs(y)) {
                exit.at(x, y) = exit.at(-std::abs(y), -std::abs(x));
                reflect_points.at(x, y) = reflect_points.at(-std::abs(y), -std::abs(x));
            } else {
                exit.at(x, y) = exit.at(-std::abs(x), -std::abs(y));
                reflect_points.at(x, y) = reflect_points.at(-std::abs(x), -std::abs(y));
            }
            continue;
        }

        Length r = std::hypot(x, y);

        if (r > exit_radius) {
            exit.at(x, y) = 0.0 * amp_unit;
            reflect_points.at(x, y) << 0.0, 0.0, 0.0;
            continue;
        }

        Angle rotation = std::atan2(y, x);

        double coef_2 = WD * WD / (a * a) + r * r / (b * b);
        double t = (coef_1 + std::sqrt(coef_1_sq + coef_0 * coef_2)) / coef_2;
        Length path_length = t * std::hypot(WD, r) + std::hypot(2.0 * f - t * WD, t * r);

        Eigen::Vector3d reflect_pos = focus_pos + Eigen::AngleAxisd(rotation, Eigen::Vector3d::UnitZ()).toRotationMatrix() * vectorize(0.0_m, t * r, -t * WD);

        if (reflect_pos(2) * 1.0_m < f - WD - ML) {
            exit.at(x, y) = 0.0 * amp_unit;
            reflect_points.at(x, y) << 0.0, 0.0, 0.0;
            continue;
        }

        reflect_points.at(x, y) = reflect_pos;
        exit.at(x, y) = std::polar(source * source_area_sqrt / path_length, k * path_length);
    }

    print_field(exit, "nodev_exit.txt");

    {
        std::ofstream file(datadir + "/reflect_points.txt");
        for (auto& x : reflect_points.line(0)) {
            for (auto& y : reflect_points.line(1)) {
                file << x.value << ' ' << y.value << ' '
                     << reflect_points.at(x, y)(0) << ' '
                     << reflect_points.at(x, y)(1) << ' '
                     << reflect_points.at(x, y)(2) << std::endl;
            }
            file << std::endl;
        }
    }

    Grid::DynamicRange<Length> focus_range{-0.5 * focus_length, 0.5 * focus_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> focus{focus_range, focus_range};
    {
        fftw_plan plan = fftw_plan_dft_2d(detector_pixel_num, detector_pixel_num,
            reinterpret_cast<fftw_complex*>(exit.data()), reinterpret_cast<fftw_complex*>(focus.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_execute(plan);

        Grid::fftshift(focus);
        print_field(focus, "nodev_focus.txt");
    }

    {
        std::ofstream file(datadir + "/profile.txt");
        for (auto& y : focus.line(1)) {
            file << y.value << ' ' << std::norm(focus.at(0.0_m, y)).value << std::endl;
        }
    }

    fftw_cleanup_threads();
    return 0;
}
