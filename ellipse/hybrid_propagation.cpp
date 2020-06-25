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
    Grid::parallelize();

    std::filesystem::create_directory(datadir);

    Grid::DynamicRange<Length> exit_range{-0.5 * exit_length, 0.5 * exit_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> exit{exit_range, exit_range};

    Grid::GridVector<Eigen::Vector3d, Length, 2> reflect_points{exit_range, exit_range};
    Grid::GridVector<Length, Length, 2> radius_map{exit_range, exit_range};
    Grid::GridVector<Length, Length, 2> path_length_map{exit_range, exit_range};

    Eigen::Vector3d source_pos = vectorize(0.0_m, 0.0_m, -f);
    Eigen::Vector3d focus_pos = vectorize(0.0_m, 0.0_m, f);

    constexpr double coef_0 = f * f / (a * a) - 1.0;
    constexpr double coef_1 = -f * WD / (a * a);
    constexpr double coef_1_sq = coef_1 * coef_1;

    std::cout << exit_radius << std::endl;
    std::cout << exit_length << std::endl;

    for (auto [x, y] : exit.lines()) {
        if (y > 0.0_m or y < x) {  // 対称性
            if (std::abs(x) < std::abs(y)) {
                exit.at(x, y) = exit.at(-std::abs(y), -std::abs(x));
                reflect_points.at(x, y) = reflect_points.at(-std::abs(y), -std::abs(x));
                reflect_points.at(x, y)(0) *= -(double)std::signbit(y);
                reflect_points.at(x, y)(1) *= -(double)std::signbit(x);
                radius_map.at(x, y) = radius_map.at(-std::abs(y), -std::abs(x));
                path_length_map.at(x, y) = path_length_map.at(-std::abs(y), -std::abs(x));
            } else {
                exit.at(x, y) = exit.at(-std::abs(x), -std::abs(y));
                reflect_points.at(x, y)(0) *= -(double)std::signbit(x);
                reflect_points.at(x, y)(1) *= -(double)std::signbit(y);
                radius_map.at(x, y) = radius_map.at(-std::abs(x), -std::abs(y));
                path_length_map.at(x, y) = path_length_map.at(-std::abs(x), -std::abs(y));
            }
            continue;
        }

        Length r = std::hypot(x, y);
        radius_map.at(x, y) = r;

        if (r > exit_radius) {
            exit.at(x, y) = 0.0 * amp_unit;
            reflect_points.at(x, y) << 0.0, 0.0, 0.0;
            radius_map.at(x, y) = 0.0_m;
            path_length_map.at(x, y) = 0.0_m;
            continue;
        }

        Angle rotation = std::atan2(y, x);

        double coef_2 = WD * WD / (a * a) + r * r / (b * b);
        double t = (-coef_1 + std::sqrt(coef_1_sq - coef_0 * coef_2)) / coef_2;
        //Length path_length = 2.0 * a - std::hypot(WD, r);
        Length path_length = std::hypot(f - t * WD - (f - WD), t * r - r) + std::hypot(-f - (f - t * WD), t * r);
        path_length_map.at(x, y) = path_length;

        //double grazing_sin = std::sqrt((1.0 + vectorize(t * WD, -t * r).dot(vectorize(2.0 * f - t * WD, -r))) / 2.0);
        exit.at(x, y) = std::polar(source * source_area_sqrt / path_length, k * path_length);

        Eigen::Vector3d reflect_pos = focus_pos + Eigen::AngleAxisd(rotation, Eigen::Vector3d::UnitZ()).toRotationMatrix() * vectorize(0.0_m, t * r, -t * WD);

        if (reflect_pos(2) * 1.0_m < f - WD - ML) {
            exit.at(x, y) = 0.0 * amp_unit;
            reflect_points.at(x, y) << 0.0, 0.0, 0.0;
            radius_map.at(x, y) = 0.0_m;
            path_length_map.at(x, y) = 0.0_m;
            continue;
        }

        reflect_points.at(x, y) = reflect_pos;
    }

    print_field(exit, "nodev_exit.txt");

    {
        std::ofstream file(datadir + "/radius.txt");
        for (auto& x : reflect_points.line(0)) {
            for (auto& y : reflect_points.line(1)) {
                file << x.value << ' ' << y.value << ' '
                     << radius_map.at(x, y).value << std::endl;
            }
            file << std::endl;
        }
    }
    {
        std::ofstream file(datadir + "/path_length.txt");
        for (auto& x : reflect_points.line(0)) {
            for (auto& y : reflect_points.line(1)) {
                file << x.value << ' ' << y.value << ' '
                     << path_length_map.at(x, y).value << std::endl;
            }
            file << std::endl;
        }
    }

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

    Grid::DynamicRange<Length> detector_range{-0.5 * detector_length, 0.5 * detector_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> detector{detector_range, detector_range};
    {
        fftw_plan plan = fftw_plan_dft_2d(detector_pixel_num, detector_pixel_num,
            reinterpret_cast<fftw_complex*>(exit.data()), reinterpret_cast<fftw_complex*>(detector.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_execute(plan);

        Grid::fftshift(detector);
        print_field(detector, "nodev_detector.txt");
    }

    {
        std::ofstream file(datadir + "/profile.txt");
        for (auto& y : detector.line(1)) {
            file << y.value << ' ' << std::norm(detector.at(0.0_m, y)).value << std::endl;
        }
    }

    return 0;
}
