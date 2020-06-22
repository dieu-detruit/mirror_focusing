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

#include <grid/algorithm.hpp>
#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>
#include <unit/double.hpp>

#include <string>
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

auto ellipse_radius(const Length& z)
{
    return b * std::sqrt(1.0 - std::pow(z / a, 2));
}
auto ellipse_normal(const Length& z)
{
    constexpr double coef = b * b / (a * a);
    return Eigen::Vector3d{0.0, 1.0, coef * z / ellipse_radius(z)}.normalized();
}

int main()
{
    Grid::parallelize();

    double NA;
    Length exit_length;

    // ミラー形状の計算と光源からの伝播
    std::vector<MicroArea> ellipsoidal_mirror;
    {
        constexpr Length dz = ML / 2400;
        constexpr Angle droll = 2.0 * M_PI / 256.0 * 1.0_rad;
        auto zrange = Grid::arange(f - ML - WD, f - WD, dz);
        auto roll_range = Grid::arange(0.0_rad, 2.0 * M_PI * 1.0_rad, droll);

        for (auto [z0, z1] : Grid::zip(zrange, Grid::shift(zrange, 1))) {
            Length z = 0.5 * (z0 + z1);
            Length radius = ellipse_radius(z);

            Area dS = 0.0_m2;
            Length _dz = dz / 8.0;
            for (auto _z : Grid::arange(z0, z1, _dz)) {
                Length _r = ellipse_radius(_z);
                dS += std::hypot(a * a * _r, b * b * _z) / (a * a) * _dz * droll;
            }
            Eigen::Vector3d normal = ellipse_normal(z);

            for (auto [roll0, roll1] : Grid::zip(roll_range, Grid::shift(roll_range, 1))) {
                Angle rotation = 0.5 * (roll0 + roll1);
                Length x = std::cos(rotation) * radius;
                Length y = std::sin(rotation) * radius;

                ellipsoidal_mirror.emplace_back(x, y, z, normal, dS);
                normal = Eigen::AngleAxis<double>(droll, Eigen::Vector3d::UnitZ()).toRotationMatrix() * normal;
            }
        }
        NA = std::sin(std::atan2(ellipse_radius(zrange.back()), WD));
        exit_length = ellipse_radius(zrange.back()) * 2.5;
        std::cout << "NA: " << NA << std::endl;
        std::cout << "Diffraction limit: " << lambda / 2.0 / NA << std::endl;
    }
    std::cout << "designed. size: " << ellipsoidal_mirror.size() << std::endl;


    constexpr Angle pitch_dev = 0.0_rad;
    constexpr Angle yaw_dev = 0.0_rad;

    std::cout << "pitch_dev: " << pitch_dev << std::endl;

    // 角度誤差を与える
    std::vector<MicroArea> installed_mirror(ellipsoidal_mirror.size());
    {
        Eigen::Vector3d origin{0.0, 0.0, rotation_center / 1.0_m};
        Eigen::Matrix3d rotation = (Eigen::AngleAxisd(yaw_dev, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(pitch_dev, Eigen::Vector3d::UnitY()))
                                       .toRotationMatrix();

        auto zip = Grid::zip(ellipsoidal_mirror, installed_mirror);
        std::for_each(std::execution::par, zip.begin(), zip.end(), [&origin, &rotation](auto x) {
            auto [ideal, installed] = x;

            installed.pos = rotation * (ideal.pos - origin) + origin;
            installed.normal = rotation * ideal.normal;
            installed.dS = ideal.dS;
        });

        //std::ofstream file("data_ellipse/dev_1mrad/installed.txt");
        //for (auto& installed : installed_mirror) {
        //file << installed.pos(0) << ' ' << installed.pos(1) << ' ' << installed.pos(2) << std::endl;
        //}
    }

    std::cout << "installed" << std::endl;

    std::vector<Complex<WaveAmplitude>> mirror_wave(installed_mirror.size());
    std::vector<Area> mirror_effective_dS(installed_mirror.size());
    // ミラーへの伝播
    {
        auto zip = Grid::zip(ellipsoidal_mirror, mirror_wave, mirror_effective_dS);
        std::for_each(std::execution::par, zip.begin(), zip.end(), [](auto x) {
            auto [area, wave, dS] = x;

            Eigen::Vector3d r = area.pos - Eigen::Vector3d{0.0, 0.0, source_z / 1.0_m};
            Length r_length = r.norm() * 1.0_m;

            wave = r(2) / (r_length / 1.0_m) / r_length
                   * std::exp(1.0i * k * r_length) * source * source_area / 1.0i / lambda;

            double grazing_sin = 1.0;
            dS = area.dS * grazing_sin;
        });
    }


    Grid::DynamicRange<Length> detector_range{-0.5 * detector_length, 0.5 * detector_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> detector{detector_range, detector_range};

    // z = f - 100nm, f, f + 100nm
    for (auto& translate : {-100.0_nm, 0.0_nm, 100.0_nm}) {
        Length knifeedge_z = f + translate;
        Length knife_focus_length = detector_pixel_num * lambda * (CD - translate) / detector_length / 2;

        Grid::DynamicRange<Length> focus_range{-0.5 * focus_length, 0.5 * focus_length, focus_pixel_num};
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> focus{focus_range, focus_range};

        focus.fill(0.0 * amp_unit);

        auto focus_zip = Grid::zip(focus.lines(), focus);
        const std::size_t size = installed_mirror.size();
        std::atomic_ulong c = 0;

        auto zip = Grid::zip(installed_mirror, mirror_wave, mirror_effective_dS);
        std::for_each(std::execution::par, zip.begin(), zip.end(), [&c, &size, &focus_zip, &knifeedge_z](auto x) {
            auto [area, wave, dS] = x;

            if (c++ % 10000 == 0) {
                std::cout << c << " / " << size << std::endl;
            }

            auto coef = wave * dS / 1.0i / lambda;
            for (auto [x, y, U] : focus_zip) {
                Eigen::Vector3d r = Eigen::Vector3d{x / 1.0_m, y / 1.0_m, knifeedge_z / 1.0_m} - area.pos;
                auto r_squared_norm = r.squaredNorm() * 1.0_m2;
                U += coef * r.dot(area.normal) * 1.0_m / r_squared_norm * std::exp(1.0i * k * std::sqrt(r_squared_norm));
            }
        });

        for (auto& x : focus.line(0)) {
            for (auto& y : focus.line(1)) {
                if (x <= 0.0_nm) {
                    focus.at(x, y) = 0.0 * amp_unit;
                }
            }
        }

        {
            std::ofstream file("data_foucaulttest_ellipse/obstructed_" + std::to_string(int(translate / 1.0_nm)) + ".txt");

            for (auto& x : focus.line(0)) {
                for (auto& y : focus.line(1)) {
                    file << x / 1.0_m << ' '
                         << y / 1.0_m << ' '
                         << std::arg(focus.at(x, y)) << ' '
                         << std::norm(focus.at(x, y)).value << std::endl;
                }
                file << std::endl;
            }
        }

        fftw_plan plan
            = fftw_plan_dft_2d(focus_pixel_num, focus_pixel_num,
                reinterpret_cast<fftw_complex*>(focus.data()), reinterpret_cast<fftw_complex*>(detector.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_execute(plan);
        Grid::fftshift(detector);

        {
            std::ofstream file("data_foucaulttest_ellipse/detected_" + std::to_string(int(translate / 1.0_nm)) + ".txt");
            for (auto& x : detector.line(0)) {
                for (auto& y : detector.line(1)) {
                    file << x / 1.0_m << ' '
                         << y / 1.0_m << ' '
                         << std::arg(detector.at(x, y)) << ' '
                         << std::norm(detector.at(x, y)).value / (detector_pixel_num * detector_pixel_num) << std::endl;
                }
                file << std::endl;
            }
        }
        fftw_destroy_plan(plan);
    }

    return 0;
}
