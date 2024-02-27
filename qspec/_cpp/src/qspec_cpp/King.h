#pragma once

#include <chrono>
#include <limits>
#include <random>
#include <set>
#include <vector>
#include <Eigen/Dense>


using namespace Eigen;


struct multivariatenormal_op
// https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
{

	multivariatenormal_op()
		: multivariatenormal_op(MatrixXd::Zero(1, 1))
	{}

	multivariatenormal_op(MatrixXd const& cov)
		: multivariatenormal_op(VectorXd::Zero(cov.rows()), cov)
	{}

	multivariatenormal_op(VectorXd  const& mean, MatrixXd  const& cov)
		: mean(mean)
	{
		LLT<MatrixXd> chol_solver(cov);
		if (chol_solver.info() == Success)
		{
			transform = chol_solver.matrixL();
		}
		else {
			SelfAdjointEigenSolver<MatrixXd> eigen_solver(cov);
			transform = eigen_solver.eigenvectors() * eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
		}
	}

	VectorXd mean;
	MatrixXd transform;

	VectorXd operator()() const
	{
		std::mt19937 gen{ std::random_device{}() };
		std::normal_distribution<double> dist;

		return mean + transform * VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
	}

	VectorXd operator()(std::mt19937& gen) const
	{
		std::normal_distribution<double> dist;

		return mean + transform * VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
	}
};


class MultivariateNormal
{

protected:
	VectorXd mean = VectorXd::Zero(1);
	MatrixXd cov = MatrixXd::Zero(1, 1);
	MatrixXd cov_inv = MatrixXd::Zero(1, 1);
	Index size = mean.size();

public:
	MultivariateNormal();
	MultivariateNormal(VectorXd _mean, MatrixXd _cov);
	~MultivariateNormal();
	VectorXd& get_mean();
	MatrixXd& get_cov();
	MatrixXd& get_cov_inv();
	Index get_size();
	multivariatenormal_op rvs{mean, cov};
	double pdf(VectorXd x);
};



class CollinearResult
{
protected:
	size_t n_samples;
	size_t n_accepted;
	std::vector<std::vector<VectorXd>> p;

public:
	CollinearResult();
	CollinearResult(size_t n, size_t size, size_t dim);
	~CollinearResult();

	size_t get_n_samples();
	size_t get_n_accepted();
	std::vector<std::vector<VectorXd>>& get_p();

	void set_n_samples(size_t n);
	void set_n_accepted(size_t n);
};

double normal_pdf(double x, double mean, double sigma);


CollinearResult collinear(std::vector<VectorXd> x, std::vector<MatrixXd> cov, size_t n, size_t n_max, unsigned int seed);
