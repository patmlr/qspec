
#include "pch.h"
#include "Results.h"


Result::~Result()
{
	std::vector<double>().swap(x);
	std::vector<VectorXd>().swap(y);
	std::vector<double>().swap(y_numpy);
}


void Result::init(std::vector<double>* _x, std::vector<VectorXd>* _y)
{
	x = *_x;
	y = *_y;
	x_size = x.size();
	y_size = y.at(0).size();
}

void Result::update()
{
	x_size = x.size();
	y_size = y.at(0).size();
	y_numpy.clear();
}

void Result::add(double _x, const VectorXd& _y)
{
	x.push_back(_x);
	y.push_back(_y);
	x_size += 1;
}

std::vector<double>* Result::get_x()
{
	return &x;
}

std::vector<VectorXd>* Result::get_y()
{
	return &y;
}

std::vector<double>* Result::get_y_numpy()
{
	if (y_numpy.size() == 0)
	{
		for (int i = 0; i < x_size; ++i)
		{
			for (int j = 0; j < y_size; ++j)
			{
				y_numpy.push_back(y.at(i)(j));
			}
		}
	}
	return &y_numpy;
}

size_t Result::get_x_size()
{
	return x_size;
}

size_t Result::get_y_size()
{
	return y_size;
}


Spectrum::Spectrum()
{
	t = std::vector<double>();
	x = std::vector<double>();
	y = std::vector<double>();
}


Spectrum::Spectrum(const std::vector<VectorXd>& delta, const std::vector<Result*>& results)
{
	std::vector<double>* _t =  results.at(0)->get_x();
	m_size = delta.at(0).size();
	t_size = _t->size();
	x_size = results.size();
	y_size = results.at(0)->get_y()->at(0).size();

	t = std::vector<double>(t_size, 0);
	x = std::vector<double>(m_size * x_size, 0);
	y = std::vector<double>(x_size * t_size * y_size, 0);
	for (size_t n_t = 0; n_t < t_size; ++n_t) t.at(n_t) = _t->at(n_t);
	for (size_t n_d = 0; n_d < x_size; ++n_d)
	{
		for (size_t m = 0; m < m_size; ++m) x.at(n_d * m_size + m) = delta.at(n_d)(m);
		for (size_t n_t = 0; n_t < t_size; ++n_t)
		{
			for (size_t i = 0; i < y_size; ++i)
			{
				y.at(n_d * t_size * y_size + n_t * y_size + i) = results.at(n_d)->get_y()->at(n_t)(i);
			}
		}
	}
}

Spectrum::~Spectrum()
{
	std::vector<double>().swap(x);
	std::vector<double>().swap(t);
	std::vector<double>().swap(y);
}

std::vector<double>* Spectrum::get_x()
{
	return &x;
}

std::vector<double>* Spectrum::get_t()
{
	return &t;
}

std::vector<double>* Spectrum::get_y()
{
	return &y;
}

size_t Spectrum::get_m_size()
{
	return m_size;
}

size_t Spectrum::get_x_size()
{
	return x_size;
}

size_t Spectrum::get_t_size()
{
	return t_size;
}

size_t Spectrum::get_y_size()
{
	return y_size;
}

void Spectrum::update()
{
	t_size = t.size();
	x_size = static_cast<size_t>(x.size() / t_size);
	y_size = static_cast<size_t>(y.size() / x_size / t_size);
}

