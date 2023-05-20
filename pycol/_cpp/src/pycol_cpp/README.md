
How to compile
=====
- include the Boost and Eigen libraries.
- Manual changes:
    * In _/boost/numeric/odeint/integrate/max_step_checker.hpp_ replace ```std::sprintf``` with ```sprintf_s```.
    * In _/boost/numeric/external/eigen/eigen_algebra.hpp_, function ```vector_space_norm_inf```,
      replace ```typedef B result_type``` with ```typedef double result_type```.
    * In _/boost/numeric/odeint/util/copy.hpp_, structure ```copy_impl_sfinae```,
      replace the line ```detail::do_copying( from , to , is_range_type() );``` by
      ```detail::do_copying( from, to, boost::mpl::false_() );```.
- Compile with >= C++ 17.
