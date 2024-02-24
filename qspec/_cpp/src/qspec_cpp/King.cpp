
#include "Physics.h"
#include "King.h"


//VectorXd straight_slope(VectorXd& p0, VectorXd& p1)
//{
//	return p1 - p0;
//}


MultivariateNormal::MultivariateNormal()
{

}

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

VectorXd& MultivariateNormal::get_mean()
{
	return mean;
}

MatrixXd& MultivariateNormal::get_cov()
{
	return cov;
}

MatrixXd& MultivariateNormal::get_cov_inv()
{
	return cov_inv;
}

Index MultivariateNormal::get_size()
{
	return size;
}

double MultivariateNormal::pdf(VectorXd x)
{
	return exp(- 0.5 * (x - mean).dot(cov_inv * (x - mean))) / sqrt(pow(2 * sc::pi, size) * cov.determinant());
}

double normal_pdf(double x, double mean, double sigma)
{
	return exp(-0.5 * pow((x - mean) / sigma, 2)) / (sqrt(2 * sc::pi) * sigma);
}

std::vector<std::vector<VectorXd>> collinear(std::vector<VectorXd> x, std::vector<MatrixXd> cov, size_t n)
{
	static std::mt19937 gen{ std::random_device{}() };
	static std::uniform_real_distribution<double> uni(0., 1.);
	std::vector<MultivariateNormal> mvn;
	for (size_t i = 0; i < x.size(); ++i) mvn.push_back(MultivariateNormal(x.at(i), cov.at(i)));


	size_t size_x2 = x.size() - 2;

	std::vector<double> u;
	std::vector<double> u_rng;
	std::vector<std::vector<VectorXd>> p;

	double u_max = 0;
	size_t mt = 0;
	size_t m = 0;
	while (m < n)
	{
		std::vector<VectorXd> _p;
		VectorXd p0 = mvn.front().rvs();
		VectorXd p1 = mvn.back().rvs();
		VectorXd dr = p1 - p0;
		dr /= dr.norm();

		double f = 1;
		double g = 1;
		_p.push_back(p0);
		for (size_t i = 0; i < size_x2; ++i)
		{
			MultivariateNormal& _mvn = mvn.at(i + 1);
			VectorXd& mean = _mvn.get_mean();
			MatrixXd& cov_inv = _mvn.get_cov_inv();
			double sigma = dr.dot(cov_inv * dr);
			double t0 = -0.5 * (dr.dot(cov_inv * (p0 - mean)) + (p0 - mean).dot(cov_inv * dr)) / sigma;
			sigma = 1 / sqrt(sigma);
			//printf("cov_inv: ((%e, %e), (%e, %e))", cov_inv(0, 0), cov_inv(0, 1), cov_inv(1, 0), cov_inv(1, 1));
			//printf("dr: (%e, %e)\n", dr(0), dr(1));
			//printf("i: %zu\n", i);
			//printf("t0: %e\n", t0);
			//printf("sigma: %e\n", sigma);
			std::normal_distribution<double> dist{ t0, sigma };
			double t = dist(gen);
			//printf("t: %e\n", t);
			//if (abs(t - t0) > 3 * sigma) t = dist(gen);
			_p.push_back(p0 + t * dr);
			//printf("p: %e, %e\n", _p.back()(0), _p.back()(1));
			f *= _mvn.pdf(_p.back());
			g *= normal_pdf(t, t0, sigma);
		}
		++mt;
		//if (t % 100000 == 0) printf("t: %zu\n", t);
		if (g < pow(1e-3, size_x2)) continue;
		_p.push_back(p1);
		double _u = f / g;
		double _u_rng = uni(gen);
		//printf("u: %e\n", _u);
		if (_u > u_max)
		{
			//printf("_u, u_max: %e, %e\n", _u, u_max);
			u_max = _u;
			m = 0;
			std::vector<bool> erase;
			bool erase_flag = false;
			for (size_t k = 0; k < u.size(); ++k)
			{
				if (u_rng.at(k) < u.at(k) / u_max)
				{
					erase.push_back(false);
					++m;
				}
				else
				{
					erase.push_back(true);
					erase_flag = true;
				}
			}
			if (erase_flag)
			{
				size_t _m = std::erase_if(u, [&u, &erase] (const double& d) {return erase.at(&d - &*u.begin()); });
				std::erase_if(u_rng, [&u_rng, &erase](const double& d) {return erase.at(&d - &*u_rng.begin()); });
				std::erase_if(p, [&p, &erase](const std::vector<VectorXd>& d) {return erase.at(&d - &*p.begin()); });
				//printf("_m: %zu\n", _m);
			}
		}
		if (_u_rng < _u / u_max)
		{
			u.push_back(_u);
			u_rng.push_back(_u_rng);
			p.push_back(_p);
			++m;
		}
		if (m % 100000 == 0) printf("\r\033[92mAccepted samples: %zu / %zu \033[0m", m, mt);
	}
	printf("\r\033[92mAccepted samples: %zu / %zu \033[0m\n", m, mt);
	printf("m: %zu\n", m);
	printf("t: %zu\n", mt);
	return p;
}


King::King(std::vector<int> _a, std::vector<double> _m, std::vector<double> _m_d, std::vector<VectorXd> _x_abs, std::vector<VectorXd> _x_abs_d)
{
	a = _a;
	m = _m;
	m_d = _m_d;
	x_abs = _x_abs;
	x_abs_d = _x_abs_d;
}

