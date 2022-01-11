#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;


class Result
{
protected:
	size_t x_size = 0;
	size_t y_size = 0;
	std::vector<double> x;
	std::vector<VectorXd> y;
	std::vector<double> y_numpy;

public:
	~Result();
	void init(std::vector<double>* _x, std::vector<VectorXd>* _y);
	std::vector<double>* get_x();
	std::vector<VectorXd>* get_y();
	std::vector<double>* get_y_numpy();
	size_t get_x_size();
	size_t get_y_size();
	void update();
	void add(double _x, const VectorXd& _y);
};


class Spectrum
{
protected:
	size_t m_size = 0;
	size_t x_size = 0;
	size_t t_size = 0;
	size_t y_size = 0;
	std::vector<double> t;
	std::vector<double> x;
	std::vector<double> y;

public:
	Spectrum();
	Spectrum(const std::vector<VectorXd>& delta, const std::vector<Result*>& results);
	~Spectrum();
	std::vector<double>* get_x();
	std::vector<double>* get_t();
	std::vector<double>* get_y();
	size_t get_m_size();
	size_t get_x_size();
	size_t get_t_size();
	size_t get_y_size();
	void update();
};
