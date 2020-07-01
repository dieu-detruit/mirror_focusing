#include <cmath>
#include <complex>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <fftw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <unit/double.hpp>
#include <unit/impl/std_overload.hpp>

#include <grid/algorithm.hpp>
#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>

#include <wavefield/core.hpp>

#include "constants.hpp"


using namespace autodiff;
using dual2nd = HigherOrderDual<2>;

std::string datadir = "data_foucaulttest";

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

    // ミラー下流開口
    Grid::DynamicRange<Length> exit_range{-0.5 * exit_length, 0.5 * exit_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> exit{exit_range, exit_range};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> exit_copy{exit_range, exit_range};
    // 焦点面
    Grid::DynamicRange<Length> focus_range{-0.5 * focus_length, 0.5 * focus_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> focus{focus_range, focus_range};
    // ディテクター面
    Grid::DynamicRange<Length> detector_range{-0.5 * detector_length, 0.5 * detector_length, detector_pixel_num};
    Grid::GridVector<Complex<WaveAmplitude>, Length, 2> detector{detector_range, detector_range};

    // 回折積分
    wavefield::FresnelFFTDiffraction exit_to_focus{exit_copy, focus, lambda};
    //wavefield::AngularSpectrumDiffraction exit_to_focus{exit_copy, focus, lambda};
    wavefield::FraunhoferDiffraction focus_to_detector{focus, detector, lambda};

    Grid::GridVector<Eigen::Vector3d, Length, 2> nodev_reflect_points{exit_range, exit_range};

    Eigen::Vector3d source_pos = vectorize(0.0_m, 0.0_m, -f);
    Eigen::Vector3d focus_pos = vectorize(0.0_m, 0.0_m, f);

    std::ofstream mirror_file(datadir + "/mirror.txt");

    std::cout << "calculating exit wavefield" << std::endl;
    {
        constexpr double coef_0 = f * f / (a * a) - 1.0;
        constexpr double coef_1 = -2.0 * f * WD / (a * a);
        constexpr double coef_1_sq = coef_1 * coef_1;

        for (auto [x, y] : exit.lines()) {
            Length r = std::hypot(x, y);

            if (r > exit_radius) {
                exit.at(x, y) = 0.0 * amp_unit;
                nodev_reflect_points.at(x, y) << 0.0, 0.0, 0.0;
                continue;
            }

            Angle rotation = std::atan2(y, x);

            double coef_2 = WD * WD / (a * a) + r * r / (b * b);
            double t = (-coef_1 + std::sqrt(coef_1_sq - 4.0 * coef_0 * coef_2)) / (2.0 * coef_2);

            Eigen::Vector3d nodev_reflect_pos = Eigen::AngleAxisd(rotation, Eigen::Vector3d::UnitZ()).toRotationMatrix() * (focus_pos + t * vectorize(r, 0.0_m, -WD));

            if (nodev_reflect_pos(2) * 1.0_m < f - WD - ML or f - WD < nodev_reflect_pos(2) * 1.0_m) {
                exit.at(x, y) = 0.0 * amp_unit;
                nodev_reflect_points.at(x, y) << 0.0, 0.0, 0.0;
                continue;
            }

            nodev_reflect_points.at(x, y) = nodev_reflect_pos;
            exit.at(x, y) = std::polar(amp_unit, -k * (x * x + y * y) / (2.0 * WD));
            mirror_file << nodev_reflect_pos(0) << ' ' << nodev_reflect_pos(1) << ' ' << nodev_reflect_pos(2) << std::endl;
        }
    }

    print_field(exit, "nodev_exit.txt");


    // 誤差なしの場合の伝播
    std::cout << "propagation from nodev mirror..." << std::endl;
    {
        std::copy(std::execution::par_unseq, exit.begin(), exit.end(), exit_copy.begin());
        exit_to_focus.propagate();
        print_field(focus, "nodev_focus.txt");

        for (auto [x, y, f] : Grid::zip(focus.lines(), focus)) {
            if (x > 0.0_m) {
                f *= 0.0;
            }
        }

        focus_to_detector.propagate();
        print_field(detector, "nodev_foucaultgram_focus.txt");
    }

    // 前にデフォーカス
    focus_range.resize(-0.5 * defocus_front_length, 0.5 * defocus_front_length, detector_pixel_num);
    focus.resize(focus_range);
    {
        std::copy(std::execution::par_unseq, exit.begin(), exit.end(), exit_copy.begin());
        exit_to_focus.propagate();

        print_field(focus, "nodev_defocus_front.txt");

        for (auto [x, y, f] : Grid::zip(focus.lines(), focus)) {
            if (x > 0.0_m) {
                f = 0.0 * amp_unit;
            }
        }

        focus_to_detector.propagate();
        print_field(detector, "nodev_foucaultgram_front.txt");
    }
    // 後ろにデフォーカス
    focus_range.resize(-0.5 * defocus_back_length, 0.5 * defocus_back_length, detector_pixel_num);
    focus.resize(focus_range);
    {
        std::copy(std::execution::par_unseq, exit.begin(), exit.end(), exit_copy.begin());
        exit_to_focus.propagate();

        print_field(focus, "nodev_defocus_back.txt");
        for (auto [x, y, f] : Grid::zip(focus.lines(), focus)) {
            if (x > 0.0_m) {
                f = 0.0 * amp_unit;
            }
        }
        focus_to_detector.propagate();
        print_field(detector, "nodev_foucaultgram_back.txt");
    }

    // 誤差を入れてみる
    constexpr Angle pitch_dev = 1.0_urad;

    Grid::GridVector<Eigen::Vector3d, Length, 2> dev_reflect_points{exit_range, exit_range};

    std::cout << "pitch deviation: " << pitch_dev << std::endl;

    Eigen::Vector3d rotation_center = vectorize(0.0_m, 0.0_m, f - WD - ML / 2.0);
    Eigen::Matrix3d dev_rotation = Eigen::AngleAxisd(pitch_dev, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d dev_rotation_inv = Eigen::AngleAxisd(-pitch_dev, Eigen::Vector3d::UnitX()).toRotationMatrix();

    Eigen::Vector3d source_pos_local = rotation_center + dev_rotation_inv * (source_pos - rotation_center);

    std::ofstream dev_mirror_file(datadir + "/dev_mirror.txt");

    int i = 0;

    for (auto [x, y] : exit.lines()) {

        if (x > 0.0_m) {  // 対称性
            exit.at(x, y) = exit.at(-x, y);
            dev_reflect_points.at(x, y) = dev_reflect_points.at(-x, y);
            dev_reflect_points.at(x, y)(0) *= -1.0;
            continue;
        }

        Eigen::Vector3d exit_pos = vectorize(x, y, f - WD);
        Eigen::Vector3d exit_pos_local = rotation_center + dev_rotation_inv * (exit_pos - rotation_center);

        // ミラー外は0
        if (nodev_reflect_points.at(x, y).norm() == 0.0) {
            exit.at(x, y) = 0.0 * amp_unit;
            dev_reflect_points.at(x, y) = Eigen::Vector3d{0.0, 0.0, 0.0};
            continue;
        }

        // ニュートン法で反射点を求める
        bool first = true;

        auto exit_deviation = [&source_pos_local, &exit_pos_local, &first](dual theta, dual phi) -> dual {
            dual sin_theta = sin(theta);
            VectorXdual pos(3);
            pos << b.value * sin_theta * cos(phi), b.value * sin_theta * sin(phi), a.value * cos(theta);

            VectorXdual normal(3);
            normal << pos(0) / (b * b).value, pos(1) / (b * b).value, pos(2) / (a * a).value;
            normal.normalize();

            VectorXdual incident_dir = source_pos_local - pos;
            VectorXdual reflection_dir = incident_dir - 2.0 * incident_dir.dot(normal) * normal;
            dual t = (reflection_dir(2) == 0.0) ? (dual)0.0 : (exit_pos_local(2) - pos(2)) / reflection_dir(2);
            VectorXdual reflected_exit_pos = pos + t * reflection_dir;

            return (reflected_exit_pos - exit_pos_local).norm();
        };

        // 初期値
        Eigen::Vector3d reflect_pos = rotation_center + dev_rotation_inv * (nodev_reflect_points.at(x, y) - rotation_center);

        dual theta, phi;
        // (x, y, z) = (b sin(theta)cos(phi), b sin(theta)sin(phi), a cos(theta))
        theta = std::acos(reflect_pos(2) / a.value);
        phi = std::atan2(reflect_pos(1), reflect_pos(0));

        Eigen::Vector3d initial_pos{
            b.value * std::sin((double)theta.val) * std::cos((double)phi.val),
            b.value * std::sin((double)theta.val) * std::sin((double)phi.val),
            a.value * std::cos((double)theta.val)};

        Length initial_path_length = ((reflect_pos - source_pos_local).norm() + (reflect_pos - exit_pos_local).norm()) * 1.0_m;

        dual D;
        for (int i = 0; i < 10; ++i) {

            D = exit_deviation(theta, phi);
            double dDdt = derivative(exit_deviation, wrt(theta), at(theta, phi));
            double dDdp = derivative(exit_deviation, wrt(phi), at(theta, phi));

            if (D.val == 0) {
                break;
            }
            if (std::abs(dDdt) < 1.0e-10 or std::abs(dDdp) < 1.0e-10) {
                break;
            }

            double alpha = dDdt * dDdt / (dDdt * dDdt + dDdp * dDdp);
            theta -= alpha * D / dDdt;
            phi -= (1.0 - alpha) * D / dDdp;
        }

        Eigen::Vector3d dev_reflect_pos_local{
            b.value * std::sin((double)theta.val) * std::cos((double)phi.val),
            b.value * std::sin((double)theta.val) * std::sin((double)phi.val),
            a.value * std::cos((double)theta.val)};


        if (dev_reflect_pos_local(2) * 1.0_m < f - WD - ML or f - WD < dev_reflect_pos_local(2) * 1.0_m) {
            exit.at(x, y) = 0.0 * amp_unit;
            dev_reflect_points.at(x, y) << 0.0, 0.0, 0.0;
            continue;
        }

        //std::cout << "⊿ z: " << reflect_pos(2) - dev_reflect_pos_local(2) << std::endl;
        double L = (dev_reflect_pos_local - source_pos_local).norm() + (dev_reflect_pos_local - exit_pos_local).norm();

        //std::cout << "⊿ L: " << L - initial_path_length / 1.0_m << std::endl;
        exit.at(x, y) = std::polar(amp_unit, -k * (x * x + y * y) / (2.0 * WD) + k * (L * 1.0_m - initial_path_length));
        dev_reflect_points.at(x, y) = rotation_center + dev_rotation * (dev_reflect_pos_local - rotation_center);

        dev_mirror_file << dev_reflect_points.at(x, y)(0) << ' ' << dev_reflect_points.at(x, y)(1) << ' ' << dev_reflect_points.at(x, y)(2) << std::endl;
    }

    print_field(exit, "dev_exit.txt");

    {
        std::ofstream file(datadir + "/dev_reflect_points.txt");
        for (auto& x : dev_reflect_points.line(0)) {
            for (auto& y : dev_reflect_points.line(1)) {
                file << x.value << ' ' << y.value << ' '
                     << dev_reflect_points.at(x, y)(0) << ' '
                     << dev_reflect_points.at(x, y)(1) << ' '
                     << dev_reflect_points.at(x, y)(2) << std::endl;
            }
            file << std::endl;
        }
    }


    // 誤差ありの場合の伝播
    // 焦点
    std::cout << "propagation from dev mirror..." << std::endl;

    focus_range.resize(-0.5 * focus_length, 0.5 * focus_length, detector_pixel_num);
    focus.resize(focus_range);
    {
        std::copy(std::execution::par_unseq, exit.begin(), exit.end(), exit_copy.begin());
        exit_to_focus.propagate();
        print_field(focus, "dev_focus.txt");

        for (auto [x, y, f] : Grid::zip(focus.lines(), focus)) {
            if (x > 0.0_m) {
                f *= 0.0;
            }
        }

        focus_to_detector.propagate();
        print_field(detector, "dev_foucaultgram_focus.txt");
    }

    // 前にデフォーカス
    focus_range.resize(-0.5 * defocus_front_length, 0.5 * defocus_front_length, detector_pixel_num);
    focus.resize(focus_range);
    {
        std::copy(std::execution::par_unseq, exit.begin(), exit.end(), exit_copy.begin());
        exit_to_focus.propagate();

        print_field(focus, "dev_defocus_front.txt");

        for (auto [x, y, f] : Grid::zip(focus.lines(), focus)) {
            if (x > 0.0_m) {
                f = 0.0 * amp_unit;
            }
        }

        focus_to_detector.propagate();
        print_field(detector, "dev_foucaultgram_front.txt");
    }
    // 後ろにデフォーカス
    focus_range.resize(-0.5 * defocus_back_length, 0.5 * defocus_back_length, detector_pixel_num);
    focus.resize(focus_range);
    {
        std::copy(std::execution::par_unseq, exit.begin(), exit.end(), exit_copy.begin());
        exit_to_focus.propagate();

        print_field(focus, "dev_defocus_back.txt");
        for (auto [x, y, f] : Grid::zip(focus.lines(), focus)) {
            if (x > 0.0_m) {
                f = 0.0 * amp_unit;
            }
        }
        focus_to_detector.propagate();
        print_field(detector, "dev_foucaultgram_back.txt");
    }

    fftw_cleanup_threads();
    return 0;
}
