#pragma once

#include <random>
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
		static std::mt19937 gen{ std::random_device{}() };
		static  std::normal_distribution<double> dist;

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
	MultivariateNormal(VectorXd _mean, MatrixXd _cov);
	~MultivariateNormal();
	Index get_size();
	multivariatenormal_op rvs{mean, cov};
	double pdf(VectorXd x);
};



class CollinearPoints
{
};


void gen_collinear(std::vector<VectorXd> x, std::vector<MatrixXd> cov, size_t n);


class KingResult
{
};


class King
{
protected:
	std::vector<int> a;
	std::vector<double> m;
	std::vector<double> m_d;
	std::vector<VectorXd> x_abs;
	std::vector<VectorXd> x_abs_d;

public:
	King(std::vector<int> _a, std::vector<double> _m, std::vector<double> _m_d, std::vector<VectorXd> _x_abs, std::vector<VectorXd> _x_abs_d);
	~King();
	KingResult fit();
	KingResult fit_nd();

};
