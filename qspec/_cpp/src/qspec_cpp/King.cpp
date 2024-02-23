
#include "Physics.h"
#include "King.h"


//VectorXd straight_slope(VectorXd& p0, VectorXd& p1)
//{
//	return p1 - p0;
//}



MultivariateNormal::MultivariateNormal(VectorXd _mean, MatrixXd _cov)
{
	mean = _mean;
	cov = _cov;
	cov_inv = cov.inverse();
	size = mean.size();
	rvs = multivariatenormal_op{mean, cov};
}

MultivariateNormal::~MultivariateNormal()
{

}

Index MultivariateNormal::get_size()
{
	return size;
}

double MultivariateNormal::pdf(VectorXd x)
{
	return exp(- 0.5 * (x - mean).dot(cov_inv * (x - mean))) / sqrt(pow(2 * sc::pi, size) * cov.determinant());
}

void gen_collinear(std::vector<VectorXd> x, std::vector<MatrixXd> cov, size_t n)
{
	static std::mt19937 gen{ std::random_device{}() };
	//static std::uniform_real_distribution<double> uni(0., 1.);
	std::vector<MultivariateNormal> mvn(x.size());
	for (size_t i = 0; i < x.size(); ++i) mvn.push_back(MultivariateNormal(x.at(i), cov.at(i)));


	size_t size_x2 = x.size() - 2;
	std::vector<double> sigma_x2(size_x2);
	for (size_t i = 0; i < size_x2; ++i) sigma_x2.push_back(cov.at(i + 1).diagonal().rowwise().maxCoeff()(0));

	size_t k = 0;
	size_t m = 0;
	while (m < n)
	{
		VectorXd p0 = mvn.at(0).rvs();
		VectorXd p1 = mvn.at(-1).rvs();

		VectorXd dr = p1 - p0;
		double dr_norm = dr.norm();

		double f = 1;
		double g = 1;
		for (size_t i = 0; i < size_x2; ++i)
		{
			double t0 = x.at(i + 1).dot(dr) / dr_norm;
			static std::normal_distribution<double> dist(t0, sigma_x2.at(i));
			double t = dist(gen);
			VectorXd p = p0 + t * dr;
			f *= mvn.at(i + 1).pdf(p);
			g *= dist(t);
		}
		double u = f / g;
		//bool a = uni(gen) < u;
		bool a = true;
		++k;
		if (not a) continue;
		++m;
	}
	printf("%zu\n", k);
	printf("%zu\n", m);
}


King::King(std::vector<int> _a, std::vector<double> _m, std::vector<double> _m_d, std::vector<VectorXd> _x_abs, std::vector<VectorXd> _x_abs_d)
{
	a = _a;
	m = _m;
	m_d = _m_d;
	x_abs = _x_abs;
	x_abs_d = _x_abs_d;
}

