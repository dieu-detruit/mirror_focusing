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

#include <unit/double.hpp>

#include "constants.hpp"

struct MicroArea {
    Eigen::Vector3d pos;
    Eigen::Vector3d normal;

    Complex<WaveAmplitude> wave;
    Area dS;

    MicroArea(Length x, Length y, Length z, Eigen::Vector3d normal, Complex<WaveAmplitude> wave, Area dS)
        : pos{x / 1.0_m, y / 1.0_m, z / 1.0_m}, normal(normal), wave(wave), dS(dS) {}
};

int main()
{
    auto parabola_z = [](const Length& d) {
        return mirror_offset_z - 0.25 * d * d / p;
    };
    auto parabola_arc_indefinite = [p = p / 1.0_m, rt_p = std::sqrt(p).value](const Length& d) -> Length {
        return 0.25 * d * std::sqrt(double(d * d / p / 1.0_m2) + 4.0) + rt_p * std::asinh(double(0.5 * d / rt_p / 1.0_m)) * 1.0_m;
    };
    auto parabola_unit_normal = [](const Length& d) {
        return Eigen::Vector3d{0.0, d / 1.0_m, -2.0 * p / 1.0_m}.normalized();
    };

    Grid::DynamicRange<Length> reflection_range{-0.5 * reflection_length, 0.5 * reflection_length, reflection_num};

    std::vector<Length> d_list(d_division);
    d_list[0] = d_min;

    std::ofstream d_file("shape2d.txt");
    for (std::size_t i = 1; i < d_division; ++i) {
        d_list[i] = d_list[i - 1] + d_coef / d_list[i - 1];
        d_file << d_list[i] << ' ' << parabola_z(d_list[i]) << std::endl;
    }

    // ミラーに到達するまでの波面
    {
        std::ofstream file("onway.txt");
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> onway{reflection_range, reflection_range};

        auto cut_coef = d_list.back() / parabola_z(d_list.back());

        for (std::size_t i = 0; i < 100; ++i) {
            Length z = i / 100.0 * mirror_offset_z;

            for (auto [x, y, o] : Grid::zip(onway.lines(), onway)) {
                Length r = std::hypot(x, y - source_d, z - source_z);
                o = z / r / r * source * std::exp(1.0i * k * r) / 1.0i / lambda * source_area;
            }

            for (auto& y : onway.line(1)) {
                if (cut_coef * z < std::abs(y) or parabola_z(y) < z) {
                    file << z / 1.0_m << ' ' << y / 1.0_m << ' ' << 0.0 << std::endl;
                    continue;
                }
                Complex<WaveAmplitude> sum = 0.0 * amp_unit;
                for (auto& x : onway.line(0)) {
                    sum += onway.at(x, y);
                }
                file << z / 1.0_m << ' ' << y / 1.0_m << ' ' << std::abs(sum) / amp_unit << std::endl;
            }
            file << std::endl;
        }
    }

    std::vector<MicroArea> parabola_mirror;


    // ミラー形状の計算と光源からの伝播
    std::ofstream shape_file("shape.txt");
    std::ofstream wave_file("wave.txt");
    std::ofstream area_file("area.txt");
    std::ofstream normal_file("normal.txt");
    for (std::size_t i = 1; i < d_division; ++i) {
        Length& d_prev = d_list[i - 1];
        Length& d = d_list[i];
        Length d_represent = 0.5 * (d - d_prev);

        Length z = parabola_z(d);
        Length r = std::hypot(d - source_d, z);

        Eigen::Vector3d normal = parabola_unit_normal(d);

        double cos = Eigen::Vector3d{(d - source_d) / 1.0_m, 0.0, z / 1.0_m}.dot(normal) / r * 1.0_m;
        double sin = std::sqrt(1.0 - cos * cos);

        Eigen::Matrix3d rotation = Eigen::AngleAxis<double>{theta_step, Eigen::Vector3d::UnitZ()}.toRotationMatrix();

        Complex<WaveAmplitude> wave = z / r / r * source * std::exp(1.0i * k * r) / 1.0i / lambda * source_area;
        Area dS = (parabola_arc_indefinite(d) - parabola_arc_indefinite(d_prev)) * theta_step * 0.5 * (d + d_prev) * sin;

        for (std::size_t j = 0; j < theta_division; ++j) {
            Angle theta = theta_step * j;

            parabola_mirror.emplace_back(d_represent * std::cos(theta), d_represent * std::sin(theta), z, normal, wave, dS);

            shape_file << z / 1.0_m << ' ' << d * std::cos(theta) / 1.0_m << ' ' << d * std::sin(theta) / 1.0_m << std::endl;
            wave_file << d * std::cos(theta) / 1.0_m << ' ' << d * std::sin(theta) / 1.0_m << ' ' << std::arg(wave) / 1.0_rad << ' ' << std::abs(wave) / amp_unit << std::endl;
            area_file << d * std::cos(theta) / 1.0_m << ' ' << d * std::sin(theta) / 1.0_m << ' ' << dS / 1.0_m2 << std::endl;
            normal_file << d * std::cos(theta) / 1.0_m << ' ' << d * std::sin(theta) / 1.0_m << ' ' << std::hypot(normal(0), normal(1)) << std::endl;

            normal = rotation * normal;
        }
    }

    // z=0での反射光
    {
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> reflection{reflection_range, reflection_range};

        reflection.fill(0.0 * amp_unit);

        int i = 0;
        for (auto [x, y, U] : Grid::zip(reflection.lines(), reflection)) {
            if (i++ % 1000 == 0) {
                std::cout << i << std::endl;
            }
            for (auto& p : parabola_mirror) {
                Eigen::Vector3d r = p.pos - Eigen::Vector3d{x / 1.0_m, y / 1.0_m, 0.039};
                Length r_length = r.norm() * 1.0_m;
                U += r.dot(p.normal) * 1.0_m / r_length / r_length * std::exp(1.0i * k * r_length) * p.wave * p.dS / 1.0i / lambda;
            }
        }

        std::ofstream file("reflection.txt");
        for (auto& x : reflection.line(0)) {
            for (auto& y : reflection.line(1)) {
                file << x / 1.0_m << ' ' << y / 1.0_m << ' '
                     << std::arg(reflection.at(x, y)) << ' '
                     << std::abs(reflection.at(x, y)) << std::endl;
            }
            file << std::endl;
        }
    }

    //z = 0までの帰り道
    {
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> return_way{reflection_range, reflection_range};

        std::ofstream file("return_way.txt");

        Length z_max = parabola_z(d_list[0]);
        for (std::size_t i = 0; i < 50; ++i) {
            std::cout << i << std::endl;
            Length z = i / 50.0 * z_max;

            return_way.fill(0.0 * amp_unit);

            for (auto [x, y, U] : Grid::zip(return_way.lines(), return_way)) {
                for (auto& p : parabola_mirror) {
                    Eigen::Vector3d r = p.pos - Eigen::Vector3d{x / 1.0_m, y / 1.0_m, z / 1.0_m};
                    Length r_length = r.norm() * 1.0_m;
                    U += r.dot(p.normal) * 1.0_m / r_length / r_length * std::exp(1.0i * k * r_length) * p.wave * p.dS / 1.0i / lambda;
                }
            }

            for (auto& y : return_way.line(1)) {
                Complex<WaveAmplitude> sum = 0.0 * amp_unit;
                for (auto& x : return_way.line(0)) {
                    sum += return_way.at(x, y);
                }
                file << z / 1.0_m << ' ' << y / 1.0_m << ' ' << std::abs(sum) / amp_unit << std::endl;
            }
            file << std::endl;
        }
    }

    return 0;
}
