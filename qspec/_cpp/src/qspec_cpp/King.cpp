
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

CollinearResult::CollinearResult()
{
	n_samples = 0;
	n_accepted = 0;
	p = std::vector<std::vector<VectorXd>>(0, std::vector<VectorXd>(0, VectorXd::Zero(0)));
}

CollinearResult::CollinearResult(size_t n, size_t size, size_t dim)
{
	n_samples = 0;
	n_accepted = 0;
	p = std::vector<std::vector<VectorXd>>(n, std::vector<VectorXd>(size, VectorXd::Zero(dim)));
}

CollinearResult::~CollinearResult()
{

}

size_t CollinearResult::get_n_samples()
{
	return n_samples;
}

size_t CollinearResult::get_n_accepted()
{
	return n_accepted;
}

std::vector<std::vector<VectorXd>>& CollinearResult::get_p()
{
	return p;
}

void CollinearResult::set_n_samples(size_t n)
{
	n_samples = n;
}

void CollinearResult::set_n_accepted(size_t n)
{
	n_accepted = n;
}

CollinearResult collinear(std::vector<VectorXd> x, std::vector<MatrixXd> cov, size_t n, size_t n_max, unsigned int seed, bool report)
{
	size_t size = x.size();
	size_t size_x2 = size - 2;
	size_t dim = x.at(0).size();

	CollinearResult res = CollinearResult(n, size, dim);

	static std::mt19937 gen{ seed };
	static std::uniform_real_distribution<double> uni(0., 1.);
	std::vector<MultivariateNormal> mvn(size);
	for (size_t i = 0; i < size; ++i) mvn.at(i) = MultivariateNormal(x.at(i), cov.at(i));

	std::vector<double> u(n);
	std::vector<double> u_rng(n);
	std::vector<std::vector<VectorXd>>& p = res.get_p();

	double time = 0;
	double u_max = 0;
	size_t index = 0;
	size_t index_max = 0;
	std::set<size_t> index_queue;
	index_queue.insert(index);
	if (n_max == 0) n_max = std::numeric_limits<size_t>::max();
	size_t n_samples = 0;
	size_t n_accepted = 0;
	while (n_accepted < n && n_samples < n_max)
	{
		index = *index_queue.begin();
		//printf("index: %zu\n", index);
		std::vector<VectorXd> _p(size);
		VectorXd p0 = mvn.front().rvs(gen);
		VectorXd p1 = mvn.back().rvs(gen);
		VectorXd dr = p1 - p0;
		dr /= dr.norm();

		double f = 1;
		double g = 1;
		_p.at(0) = p0;
		for (size_t i = 0; i < size_x2; ++i)
		{
			MultivariateNormal& _mvn = mvn.at(i + 1);
			VectorXd& mean = _mvn.get_mean();
			MatrixXd& cov_inv = _mvn.get_cov_inv();
			VectorXd a = cov_inv * dr;
			double sigma = dr.dot(a);
			double t0 = -(p0 - mean).dot(a) / sigma;
			sigma = 1 / sqrt(sigma);
			//printf("cov_inv: ((%e, %e), (%e, %e))\n", cov_inv(0, 0), cov_inv(0, 1), cov_inv(1, 0), cov_inv(1, 1));
			//printf("dr: (%e, %e)\n", dr(0), dr(1));
			//printf("i: %zu\n", i);
			//printf("t0: %e\n", t0);
			//printf("sigma: %e\n", sigma);
			std::normal_distribution<double> dist{ t0, sigma };
			double t = dist(gen);
			//printf("t: %e\n", t);
			//if (abs(t - t0) > 3 * sigma) t = dist(gen);
			_p.at(i + 1) = p0 + t * dr;
			//printf("p: %e, %e\n", _p.back()(0), _p.back()(1));
			f *= _mvn.pdf(_p.at(i + 1));
			g *= normal_pdf(t, t0, sigma);
		}
		++n_samples;
		//if (t % 100000 == 0) printf("t: %zu\n", t);
		//if (g < pow(1e-3, size_x2)) continue;
		_p.at(size - 1) = p1;
		double _u = f / g;
		double _u_rng = uni(gen);
		//printf("1 _u, u_max: %e, %e\n", _u, u_max);
		// std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		if (_u > u_max)
		{
			//printf("2 _u, u_max: %e, %e\n", _u, u_max);
			u_max = _u;
			for (size_t k = 0; k < index_max; ++k)
			{
				if (u_rng.at(k) >= u.at(k) / u_max)
				{
					if (index_queue.insert(k).second) n_accepted -= 1;
					//printf("k: %zu\n", k);
				}
			}
		}
		if (_u_rng < _u / u_max)
		{
			u.at(index) = _u;
			u_rng.at(index) = _u_rng;
			p.at(index).swap(_p);
			index_queue.erase(index);
			++n_accepted;
			if (index_queue.empty())
			{
				if (++index_max < n) index_queue.insert(index_max);
			}
		}
		// std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		// time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
		if (report && (n_accepted + n_samples) % 100000 == 0) printf("\r\033[92mAccepted samples: %zu / %zu \033[0m", n_accepted, n_samples);
	}
	if (report) printf("\r\033[92mAccepted samples: %zu / %zu \033[0m\n", n_accepted, n_samples);
	// printf("Time difference (sec) = %1.3f", time);

	res.set_n_samples(n_samples);
	res.set_n_accepted(n_accepted);
	return res;
}
