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

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <unit/double.hpp>
#include <unit/impl/std_overload.hpp>

#include <wavefield/core.hpp>

#include <grid/algorithm.hpp>
#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>

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

auto ellipse_radius(Length z)
{
    return std::sqrt(a * a - z * z) * b / a;
}
auto ellipse_radius(double z)
{
    return std::sqrt(a.value * a.value - z * z) * b.value / a.value;
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

    constexpr Length focus_length = 1000.0_nm;
    constexpr Length exit_length = focus_pixel_num * lambda * WD / focus_length;


    Grid::DynamicRange<Length> exit_range{-0.5 * exit_length, 0.5 * exit_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> exit{exit_range, exit_range};

    Grid::GridVector<Eigen::Vector3d, Length, 2> nodev_reflect_points{exit_range, exit_range};

    Eigen::Vector3d source_pos = vectorize(0.0_m, 0.0_m, -f);
    Eigen::Vector3d focus_pos = vectorize(0.0_m, 0.0_m, f);

    std::cout << "calculating exit wavefield" << std::endl;
    {
        constexpr double coef_0 = 1.0 - f * f / (a * a);
        constexpr double coef_1 = f * WD / (a * a);
        constexpr double coef_1_sq = coef_1 * coef_1;

        std::cout << exit_radius << std::endl;
        std::cout << exit_length << std::endl;

        for (auto [x, y] : exit.lines()) {
            if (y > 0.0_m or y < x) {  // 対称性
                if (std::abs(x) < std::abs(y)) {
                    exit.at(x, y) = exit.at(-std::abs(y), -std::abs(x));
                    nodev_reflect_points.at(x, y) = nodev_reflect_points.at(-std::abs(y), -std::abs(x));
                    nodev_reflect_points.at(x, y)(0) *= std::signbit(y) ? -1.0 : 1.0;
                    nodev_reflect_points.at(x, y)(1) *= std::signbit(x) ? -1.0 : 1.0;
                } else {
                    exit.at(x, y) = exit.at(-std::abs(x), -std::abs(y));
                    nodev_reflect_points.at(x, y) = nodev_reflect_points.at(-std::abs(x), -std::abs(y));
                    nodev_reflect_points.at(x, y)(0) *= std::signbit(x) ? -1.0 : 1.0;
                    nodev_reflect_points.at(x, y)(1) *= std::signbit(y) ? -1.0 : 1.0;
                }
                continue;
            }

            Length r = std::hypot(x, y);

            if (r > exit_radius) {
                exit.at(x, y) = 0.0 * amp_unit;
                nodev_reflect_points.at(x, y) << 0.0, 0.0, 0.0;
                continue;
            }

            Angle rotation = std::atan2(y, x);

            double coef_2 = WD * WD / (a * a) + r * r / (b * b);
            double t = (coef_1 + std::sqrt(coef_1_sq + coef_0 * coef_2)) / coef_2;
            Length path_length = t * std::hypot(WD, r) + std::hypot(2.0 * f - t * WD, t * r);

            Eigen::Vector3d nodev_reflect_pos = focus_pos + Eigen::AngleAxisd(rotation, Eigen::Vector3d::UnitZ()).toRotationMatrix() * vectorize(0.0_m, t * r, -t * WD);

            if (nodev_reflect_pos(2) * 1.0_m < f - WD - ML) {
                exit.at(x, y) = 0.0 * amp_unit;
                nodev_reflect_points.at(x, y) << 0.0, 0.0, 0.0;
                continue;
            }

            nodev_reflect_points.at(x, y) = nodev_reflect_pos;
            exit.at(x, y) = amp_unit;
        }
    }

    print_field(exit, "nodev_exit.txt");

    {
        std::ofstream file(datadir + "/nodev_reflect_points.txt");
        for (auto& x : nodev_reflect_points.line(0)) {
            for (auto& y : nodev_reflect_points.line(1)) {
                file << x.value << ' ' << y.value << ' '
                     << nodev_reflect_points.at(x, y)(0) << ' '
                     << nodev_reflect_points.at(x, y)(1) << ' '
                     << nodev_reflect_points.at(x, y)(2) << std::endl;
            }
            file << std::endl;
        }
    }

    // 誤差なしの場合の伝播
    std::cout << "propagation from nodev mirror..." << std::endl;
    {
        Grid::DynamicRange<Length> focus_range{-0.5 * focus_length, 0.5 * focus_length, detector_pixel_num};
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> focus{focus_range, focus_range};
        {
            fftw_plan plan = fftw_plan_dft_2d(detector_pixel_num, detector_pixel_num,
                reinterpret_cast<fftw_complex*>(exit.data()), reinterpret_cast<fftw_complex*>(focus.data()), FFTW_FORWARD, FFTW_ESTIMATE);

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
    }

    // 誤差を入れてみる
    constexpr Angle pitch_dev = 0.1_urad;

    std::cout << "pitch deviation: " << pitch_dev << std::endl;

    Eigen::Vector3d rotation_center = vectorize(0.0_m, 0.0_m, f - WD - ML / 2.0);
    Eigen::Matrix3d dev_rotation = Eigen::AngleAxisd(pitch_dev, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d dev_rotation_inv = Eigen::AngleAxisd(-pitch_dev, Eigen::Vector3d::UnitX()).toRotationMatrix();

    Eigen::Vector3d source_pos_local = rotation_center + dev_rotation_inv * (source_pos - rotation_center);

    using namespace autodiff;

    int i = 0;
    for (auto [x, y] : exit.lines()) {

        if (x > 0.0_m) {  // 対称性
            exit.at(x, y) = exit.at(-x, y);
            continue;
        }

        Eigen::Vector3d exit_pos = vectorize(x, y, f - WD);
        Eigen::Vector3d exit_pos_local = rotation_center + dev_rotation_inv * (exit_pos - rotation_center);

        // ミラー外は0
        if (nodev_reflect_points.at(x, y).norm() == 0.0) {
            exit.at(x, y) = 0.0 * amp_unit;
            continue;
        }

        if (i++ > 100) {
            break;
        }

        // 誤差なしの時の反射点を初期値に取る
        Eigen::Vector3d reflect_pos = nodev_reflect_points.at(x, y);

        double initial_cos_theta = reflect_pos(2) / a.value;
        double initial_sin_theta = std::sqrt(1.0 - initial_cos_theta * initial_cos_theta);

        using dual2nd = HigherOrderDual<2>;

        dual2nd theta = std::acos(initial_cos_theta);
        dual2nd phi = std::acos(reflect_pos(0) / b.value / initial_sin_theta);

        std::cout << "i: " << i << ", initial theta: " << theta << ", phi: " << phi << std::endl;

        auto path_length = [&source_pos_local, &exit_pos_local](dual2nd theta, dual2nd phi) -> dual2nd {
            VectorXdual2nd pos(3);
            dual2nd sin_theta = sin(theta);
            pos << b.value * sin_theta * cos(phi), b.value * sin_theta * sin(phi), a.value * cos(theta);
            return (pos - source_pos_local).norm() + (pos - exit_pos_local).norm();
        };

        Length initial_path_length = (double)path_length(theta, phi).val * 1.0_m;

        dual2nd L;
        for (int j = 0; j < 500; ++j) {
            dual2nd sin_theta = sin(theta);
            dual2nd cos_theta = cos(theta);
            dual2nd cos_phi = cos(phi);
            dual2nd sin_phi = sin(phi);

            L = path_length(theta, phi);

            double dL_dtheta = derivative(path_length, wrt(theta), forward::at(theta, phi)).val;
            double d2L_dtheta2 = derivative(path_length, wrt<2>(theta), forward::at(theta, phi));
            double dL_dphi = derivative(path_length, wrt(phi), forward::at(theta, phi)).val;
            double d2L_dphi2 = derivative(path_length, wrt<2>(phi), forward::at(theta, phi));

            if (std::abs(dL_dtheta) and std::abs(dL_dphi) < 1.0e-7) {
                break;
            }
            if (d2L_dtheta2 == 0.0 and d2L_dphi2 == 0.0) {
                break;
            }

            theta.val -= dL_dtheta / d2L_dtheta2;
            phi.val -= dL_dphi / d2L_dphi2;

            std::cout << "i: " << i << ", j: " << j << ",L: " << L << ", dL/dth: " << dL_dtheta << ", dL/dph: " << dL_dphi
                      << ", theta: " << theta << ", phi: " << phi << std::endl;

            exit.at(x, y) = std::polar(amp_unit, k * ((double)L.val * 1.0_m - initial_path_length));
        }
    }

    // 誤差アリの場合の伝播
    {
        Grid::DynamicRange<Length> focus_range{-0.5 * focus_length, 0.5 * focus_length, detector_pixel_num};
        Grid::GridVector<Complex<WaveAmplitude>, Length, 2> focus{focus_range, focus_range};
        {
            fftw_plan plan = fftw_plan_dft_2d(detector_pixel_num, detector_pixel_num,
                reinterpret_cast<fftw_complex*>(exit.data()), reinterpret_cast<fftw_complex*>(focus.data()), FFTW_FORWARD, FFTW_ESTIMATE);

            fftw_execute(plan);

            Grid::fftshift(focus);
            print_field(focus, "dev_focus.txt");
        }

        {
            std::ofstream file(datadir + "/dev_profile.txt");
            for (auto& y : focus.line(1)) {
                file << y.value << ' ' << std::norm(focus.at(0.0_m, y)).value << std::endl;
            }
        }
    }

    fftw_cleanup_threads();
    return 0;
}
