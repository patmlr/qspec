#pragma once

#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>
#include <Eigen/Dense>

namespace boost {
namespace numeric {
namespace odeint {

template<>
struct vector_space_norm_inf<Eigen::VectorXcd>
{
    using result_type = double;

    double operator()(const Eigen::VectorXcd& x) const
    {
        return x.cwiseAbs().maxCoeff();
    }
};

template<>
struct vector_space_norm_inf<Eigen::MatrixXcd>
{
    using result_type = double;

    double operator()(const Eigen::MatrixXcd& x) const
    {
        return x.cwiseAbs().maxCoeff();
    }
};

}
}
}