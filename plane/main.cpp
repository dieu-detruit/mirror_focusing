#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <execution>
#include <fstream>
#include <iostream>

#include <fftw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>
#include <unit/double.hpp>

#include <unit/double.hpp>
#include <vector>

#include "constants.hpp"

struct MicroArea {
    Eigen::Vector3d pos;
    Eigen::Vector3d normal;

    Area dS;

    MicroArea(Length x, Length y, Length z, Eigen::Vector3d normal, Area dS)
        : pos{x / 1.0_m, y / 1.0_m, z / 1.0_m}, normal(normal), dS(dS) {}
    MicroArea()
        : pos(Eigen::Vector3d::Zero()), normal(Eigen::Vector3d::Zero()), dS(0.0_m2) {}
};

int main()
{
    // ミラー形状
    std::vector<MicroArea> plane_mirror;
    {
        constexpr Length dz = 0.01_mm;
        auto zrange = Grid::arange(0.5 * (-ML + dz), 0.5 * (ML - dz), dz);

        Area dS = dz * dz;
        Eigen::Vector3d normal{0.0, 1.0, 0.0};
        for (auto [z, x] : Grid::prod(zrange, zrange)) {
            plane_mirror.emplace_back(x, 0.0_mm, z, normal, dS);
        }
    }
    std::cout << "designed. size: " << plane_mirror.size() << std::endl;

    constexpr Angle pitch_dev = 0.0 * M_PI * 1.0_rad;
    constexpr Angle yaw_dev = 0.0 * M_PI * 1.0_rad;

    // 角度誤差を与える
    std::vector<MicroArea> installed_mirror(plane_mirror.size());
    {
        Eigen::Matrix3d rotation = (Eigen::AngleAxisd(yaw_dev, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(pitch_dev, Eigen::Vector3d::UnitY()))
                                       .toRotationMatrix();

        for (auto [ideal, installed] : Grid::zip(plane_mirror, installed_mirror)) {
            installed.pos = rotation * ideal.pos;
            installed.normal = rotation * ideal.normal;
            installed.dS = ideal.dS;
        }

        std::ofstream file("data_plane/installed.txt");
        for (auto& installed : installed_mirror) {
            file << installed.pos(0) << ' ' << installed.pos(1) << ' ' << installed.pos(2) << std::endl;
        }
    }
    {
        std::ofstream file("data_plane/dS.txt");
        for (auto& installed : installed_mirror) {
            file << installed.pos(0) << ' ' << installed.pos(1) << ' ' << installed.dS << std::endl;
        }
    }
    {
        std::ofstream file("data_plane/normal.txt");
        for (auto& installed : installed_mirror) {
            file << installed.pos(0) << ' ' << installed.pos(1) << ' ' << installed.normal(0) << ' ' << installed.normal(1) << ' ' << installed.normal(2) << std::endl;
        }
    }

    std::cout << "installed" << std::endl;

    std::vector<Complex<WaveAmplitude>> mirror_wave(installed_mirror.size());
    std::vector<Area> mirror_effective_dS(installed_mirror.size());
    // ミラーへの伝播
    {
        for (auto [area, wave, dS] : Grid::zip(installed_mirror, mirror_wave, mirror_effective_dS)) {
            Eigen::Vector3d r = area.pos - Eigen::Vector3d{0.0, source_y / 1.0_m, source_z / 1.0_m};
            Length r_length = r.norm() * 1.0_m;

            wave = r(2) / (r_length / 1.0_m) / r_length
                   * std::exp(1.0i * k * r_length) * source * source_area / 1.0i / lambda;

            double grazing_sin = 1.0;
            dS = area.dS * grazing_sin;
        }
    }

    {
        std::ofstream file("data_plane/effective_dS.txt");
        for (auto& dS : mirror_effective_dS) {
            file << dS << std::endl;
        }
    }

    // z=fでの反射光
    {
        Grid::DynamicRange<Length> reflection_range{-0.5 * reflection_length, 0.5 * reflection_length, reflection_pixel_num};
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> reflection{reflection_range, reflection_range};
        reflection.fill(0.0 * amp_unit);

        auto reflection_zip = Grid::zip(reflection.lines(), reflection);
        const std::size_t size = installed_mirror.size();
        std::atomic_ulong c = 0;

        auto zip = Grid::zip(installed_mirror, mirror_wave, mirror_effective_dS);
        std::for_each(std::execution::par, zip.begin(), zip.end(), [&c, &size, &reflection_zip](auto x) {
            auto [area, wave, dS] = x;

            if (c++ % 1000 == 0) {
                std::cout << c << " / " << size << std::endl;
            }

            auto coef = wave * dS / 1.0i / lambda;
            for (auto [x, y, U] : reflection_zip) {
                Eigen::Vector3d r = Eigen::Vector3d{x / 1.0_m, y / 1.0_m, reflection_z / 1.0_m} - area.pos;
                Length r_length = r.norm() * 1.0_m;
                U += coef * r.dot(area.normal) * 1.0_m / (r_length * r_length) * std::exp(1.0i * k * r_length);
            }
        });

        {
            std::ofstream file("data_plane/reflection.txt");
            for (auto& x : reflection.line(0)) {
                for (auto& y : reflection.line(1)) {
                    file << x / 1.0_m << ' ' << y / 1.0_m << ' '
                         << std::arg(reflection.at(x, y)) << ' '
                         << std::abs(reflection.at(x, y)).value << std::endl;
                }
                file << std::endl;
            }
        }
        {
            std::ofstream file("data_plane/reflection_xeq0.txt");
            for (auto& x : reflection.line(0)) {
                file << x / 1.0_m << ' '
                     << std::arg(reflection.at(x, 0.0_mm)) << ' '
                     << std::abs(reflection.at(x, 0.0_mm)).value << std::endl;
            }
            file << std::endl;
        }

        Length exit_length = 100.0_mm;
        Grid::DynamicRange<Length> exit_range{-0.5 * exit_length, 0.5 * exit_length, reflection_pixel_num};
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> exit{exit_range, exit_range};

        fftw_plan plan
            = fftw_plan_dft_2d(reflection_pixel_num, reflection_pixel_num,
                reinterpret_cast<fftw_complex*>(reflection.data()), reinterpret_cast<fftw_complex*>(exit.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_execute(plan);
        {
            std::ofstream file("data_plane/exit.txt");

            for (auto& x : exit.line(0)) {
                for (auto& y : exit.line(1)) {
                    file << x / 1.0_m << ' '
                         << y / 1.0_m << ' '
                         << std::arg(exit.at(x, y)) << ' '
                         << std::abs(exit.at(x, y)).value << std::endl;
                }
                file << std::endl;
            }
        }
        fftw_destroy_plan(plan);
    }

    return 0;
}
