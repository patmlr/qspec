
How to compile:
    - include the Boost and Eigen libraries.
    - Manual changes:
        * In "/boost/numeric/odeint/integrate/max_step_checker.hpp" replace 'std::sprintf' by 'sprintf_s'.
        * In "/boost/numeric/external/eigen/eigen_algebra.hpp", function 'vector_space_norm_inf',
          replace 'typedef B result_type' by 'typedef double result_type'.
        * In "/boost/numeric/odeint/util/copy.hpp", struct 'copy_impl_sfinae',
          replace the line 'detail::do_copying( from , to , is_range_type() );' by
          'detail::do_copying(from, to, boost::mpl::false_());'.
    - Compile with >= C++ 17.
