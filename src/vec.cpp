// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "vec.h"
#include "rand.h"
#include "error.h"
#include "matrix.h"
#include "string.h"
#include "json.h"
#include <cmath>
#include <cstdio>
#include <cstring>


using std::vector;



Vec::Vec(size_t n)
: m_size(n)
{
	if(n == 0)
		m_data = NULL;
	else
		m_data = new double[n];
}

Vec::Vec(int n)
: m_size(n)
{
	if(n == 0)
		m_data = NULL;
	else
		m_data = new double[n];
}

Vec::Vec(const std::initializer_list<double>& list)
: m_size(list.size())
{
	if(list.size() == 0)
		m_data = nullptr;
	else
		m_data = new double[list.size()];
	size_t i = 0;
	for(const double* it = begin(list); it != end(list); ++it)
	{
		m_data[i++] = *it;
	}
}

Vec::Vec(double d)
{
	throw Ex("Calling this method is an error");
}

Vec::Vec(const Vec& orig)
{
	m_size = orig.m_size;
	if(m_size == 0)
		m_data = NULL;
	else
	{
		m_data = new double[m_size];
		for(size_t i = 0; i < m_size; i++)
			m_data[i] = orig.m_data[i];
	}
}

Vec::Vec(const JsonNode& node)
{
	JsonListIterator it(&node);
	m_size = it.remaining();
	if(m_size == 0)
		m_data = NULL;
	else
		m_data = new double[m_size];
	size_t i = 0;
	while(it.remaining() > 0)
	{
		m_data[i++] = it.currentDouble();
		it.advance();
	}
}

Vec::~Vec()
{
	delete[] m_data;
}

Vec& Vec::operator=(const Vec& orig)
{
	throw Ex("Call copy instead to avoid confusion");
}

JsonNode* Vec::marshal(Json& doc)
{
	JsonNode* pList = doc.newList();
	for(size_t i = 0; i < m_size; i++)
		pList->add(&doc, m_data[i]);
	return pList;
}

void Vec::copy(const Vec& orig)
{
	resize(orig.m_size);
	for(size_t i = 0; i < m_size; i++)
		m_data[i] = orig.m_data[i];
}

void Vec::resize(size_t n)
{
	if(m_size == n)
		return;
	delete[] m_data;
	m_size = n;
	if(n == 0)
		m_data = NULL;
	else
		m_data = new double[n];
}

void Vec::fill(const double val, size_t startPos, size_t endPos)
{
	endPos = std::min(endPos, m_size);
	for(size_t i = startPos; i < endPos; i++)
		m_data[i] = val;
}

Vec Vec::operator+(const Vec& that) const
{
	Vec v(m_size);
	for(size_t i = 0; i < m_size; i++)
		v[i] = (*this)[i] + that[i];
	return v;
}

Vec& Vec::operator+=(const Vec& that)
{
	for(size_t i = 0; i < m_size; i++)
		(*this)[i] += that[i];
	return *this;
}

Vec Vec::operator-(const Vec& that) const
{
	Vec v(m_size);
	for(size_t i = 0; i < m_size; i++)
		v[i] = (*this)[i] - that[i];
	return v;
}

Vec& Vec::operator-=(const Vec& that)
{
	for(size_t i = 0; i < m_size; i++)
		(*this)[i] -= that[i];
	return *this;
}

Vec Vec::operator*(double scalar) const
{
	Vec v(m_size);
	for(size_t i = 0; i < m_size; i++)
		v[i] = (*this)[i] * scalar;
	return v;
}

Vec& Vec::operator*=(double scalar)
{
	for(size_t i = 0; i < m_size; i++)
		(*this)[i] *= scalar;
	return *this;
}

void Vec::set(const double* pSource, size_t n)
{
	resize(n);
	for(size_t i = 0; i < n; i++)
		(*this)[i] = *(pSource++);
}

double Vec::squaredMagnitude() const
{
	double s = 0.0;
	for(size_t i = 0; i < m_size; i++)
	{
		double d = (*this)[i];
		s += (d * d);
	}
	return s;
}

void Vec::normalize()
{
	double mag = std::sqrt(squaredMagnitude());
	if(mag < 1e-16)
		fill(std::sqrt(1.0 / m_size));
	else
		(*this) *= (1.0 / mag);
}

double Vec::squaredDistance(const Vec& that) const
{
	double s = 0.0;
	for(size_t i = 0; i < m_size; i++)
	{
		double d = (*this)[i] - that[i];
		s += (d * d);
	}
	return s;
}

void Vec::fillUniform(Rand& rand, double min, double max)
{
	for(size_t i = 0; i < m_size; i++)
		(*this)[i] = rand.uniform() * (max - min) + min;
}

void Vec::fillNormal(Rand& rand, double deviation)
{
	for(size_t i = 0; i < m_size; i++)
		(*this)[i] = rand.normal() * deviation;
}

void Vec::fillSphericalShell(Rand& rand, double radius)
{
	fillNormal(rand);
	normalize();
	if(radius != 1.0)
		(*this) *= radius;
}

void Vec::fillSphericalVolume(Rand& rand)
{
	fillSphericalShell(rand);
	(*this) *= std::pow(rand.uniform(), 1.0 / m_size);
}

void Vec::fillSimplex(Rand& rand)
{
	for(size_t i = 0; i < m_size; i++)
		(*this)[i] = rand.exponential();
	(*this) *= (1.0 / sum());
}

void Vec::print(std::ostream& stream) const
{
	stream << "[";
	if(m_size > 0)
		stream << to_str((*this)[0]);
	for(size_t i = 1; i < m_size; i++)
		stream << "," << to_str((*this)[i]);
	stream << "]";
}

double Vec::sum() const
{
	double s = 0.0;
	for(size_t i = 0; i < m_size; i++)
		s += (*this)[i];
	return s;
}

size_t Vec::indexOfMax(size_t startPos, size_t endPos) const
{
	endPos = std::min(m_size, endPos);
	size_t maxIndex = startPos;
	double maxValue = -1e300;
	for(size_t i = startPos; i < endPos; i++)
	{
		if((*this)[i] > maxValue)
		{
			maxIndex = i;
			maxValue = (*this)[i];
		}
	}
	return maxIndex;
}

double Vec::dotProduct(const Vec& that) const
{
	double s = 0.0;
	for(size_t i = 0; i < m_size; i++)
		s += ((*this)[i] * that[i]);
	return s;
}

double Vec::dotProductIgnoringUnknowns(const Vec& that) const
{
	double s = 0.0;
	for(size_t i = 0; i < m_size; i++)
	{
		if((*this)[i] != UNKNOWN_VALUE && that[i] != UNKNOWN_VALUE)
			s += ((*this)[i] * that[i]);
	}
	return s;
}

double Vec::estimateSquaredDistanceWithUnknowns(const Vec& that) const
{
	double dist = 0;
	double d;
	size_t nMissing = 0;
	for(size_t n = 0; n < m_size; n++)
	{
		if((*this)[n] == UNKNOWN_VALUE || that[n] == UNKNOWN_VALUE)
			nMissing++;
		else
		{
			d = (*this)[n] - that[n];
			dist += (d * d);
		}
	}
	if(nMissing >= m_size)
		return 1e50; // we have no info, so let's make a wild guess
	else
		return dist * m_size / (m_size - nMissing);
}

void Vec::addScaled(double scalar, const Vec& that)
{
	for(size_t i = 0; i < m_size; i++)
		(*this)[i] += (scalar * that[i]);
}

void Vec::regularize_L1(double amount)
{
	for(size_t i = 0; i < m_size; i++)
	{
		if((*this)[i] < 0.0)
			(*this)[i] = std::min(0.0, (*this)[i] + amount);
		else
			(*this)[i] = std::max(0.0, (*this)[i] - amount);
	}
}

void Vec::put(size_t pos, const Vec& that, size_t start, size_t length)
{
	if(length == (size_t)-1)
		length = that.size() - start;
	else if(start + length > that.size())
		throw Ex("Input out of range. that size=", to_str(that.size()), ", start=", to_str(start), ", length=", to_str(length));
	if(pos + length > m_size)
		throw Ex("Out of range. this size=", to_str(m_size), ", pos=", to_str(pos), ", that size=", to_str(that.m_size));
	for(size_t i = 0; i < length; i++)
		(*this)[pos + i] = that[start + i];
}

void Vec::erase(size_t start, size_t count)
{
	if(start + count > m_size)
		throw Ex("out of range");
	size_t end = m_size - count;
	for(size_t i = start; i < end; i++)
		(*this)[i] = (*this)[i + count];
	m_size -= count;
}

double Vec::correlation(const Vec& that) const
{
	double d = this->dotProduct(that);
	if(d == 0.0)
		return 0.0;
	return d / (sqrt(this->squaredMagnitude() * that.squaredMagnitude()));
}




std::string to_str(const Vec& v)
{
	std::ostringstream os;
	if(v.size() > 0)
		os << to_str(v[0]);
	for(size_t i = 1; i < v.size(); i++)
		os << "," << to_str(v[i]);
	return os.str();
}











size_t Tensor_countTensorSize(size_t dimCount, const size_t* dims)
{
	size_t n = 1;
	for(size_t i = 0; i < dimCount; i++)
		n *= dims[i];
	return n;
}

Tensor::Tensor(Vec& vals, const std::initializer_list<size_t>& list)
: VecWrapper(vals)
{
	dimCount = list.size();
	dims = new size_t[dimCount];
	size_t i = 0;
	size_t tot = 1;
	for(const size_t* it = begin(list); it != end(list); ++it)
	{
		dims[i++] = *it;
		tot *= *it;
	}
	if(tot != vals.size())
		throw Ex("Mismatching sizes. Vec has ", to_str(vals.size()), ", Tensor has ", to_str(tot));
}

Tensor::Tensor(const Tensor& copyMe)
: VecWrapper(copyMe.m_data, copyMe.m_size)
{
	dimCount = copyMe.dimCount;
	dims = new size_t[dimCount];
	for(size_t i = 0; i < dimCount; i++)
		dims[i] = copyMe.dims[i];
}

Tensor::Tensor(double* buf, size_t _dimCount, const size_t* _dims)
: VecWrapper(buf, buf ? Tensor_countTensorSize(_dimCount, _dims) : 0)
{
	dimCount = _dimCount;
	dims = new size_t[dimCount];
	for(size_t i = 0; i < dimCount; i++)
		dims[i] = _dims[i];
}

Tensor::~Tensor()
{
	delete[] dims;
}

void Tensor::convolve(const Tensor& in, const Tensor& filter, Tensor& out, bool flipFilter, size_t stride)
{
	// Precompute some values
	size_t dc = in.dimCount;
	if(dc != filter.dimCount)
		throw Ex("Expected tensors with the same number of dimensions");
	if(dc != out.dimCount)
		throw Ex("Expected tensors with the same number of dimensions");
	size_t* kinner = (size_t*)alloca(sizeof(size_t) * 5 * dc);
	size_t* kouter = kinner + dc;
	size_t* stepInner = kouter + dc;
	size_t* stepFilter = stepInner + dc;
	size_t* stepOuter = stepFilter + dc;

	// Compute step sizes
	stepInner[0] = 1;
	stepFilter[0] = 1;
	stepOuter[0] = 1;
	for(size_t i = 1; i < dc; i++)
	{
		stepInner[i] = stepInner[i - 1] * in.dims[i - 1];
		stepFilter[i] = stepFilter[i - 1] * filter.dims[i - 1];
		stepOuter[i] = stepOuter[i - 1] * out.dims[i - 1];
	}
	size_t filterTail = stepFilter[dc - 1] * filter.dims[dc - 1] - 1;

	// Do convolution
	size_t op = 0;
	size_t ip = 0;
	size_t fp = 0;
	for(size_t i = 0; i < dc; i++)
	{
		kouter[i] = 0;
		kinner[i] = 0;
		ssize_t padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
		ssize_t adj = (padding - std::min(padding, (ssize_t)kouter[i])) - kinner[i];
		kinner[i] += adj;
		fp += adj * stepFilter[i];
	}
	while(true) // kouter
	{
		double val = 0.0;

		// Fix up the initial kinner positions
		for(size_t i = 0; i < dc; i++)
		{
			ssize_t padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
			ssize_t adj = (padding - std::min(padding, (ssize_t)kouter[i])) - kinner[i];
			kinner[i] += adj;
			fp += adj * stepFilter[i];
			ip += adj * stepInner[i];
		}
		while(true) // kinner
		{
			val += (in[ip] * filter[flipFilter ? filterTail - fp : fp]);

			// increment the kinner position
			size_t i;
			for(i = 0; i < dc; i++)
			{
				kinner[i]++;
				ip += stepInner[i];
				fp += stepFilter[i];
				ssize_t padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
				if(kinner[i] < filter.dims[i] && kouter[i] + kinner[i] - padding < in.dims[i])
					break;
				ssize_t adj = (padding - std::min(padding, (ssize_t)kouter[i])) - kinner[i];
				kinner[i] += adj;
				fp += adj * stepFilter[i];
				ip += adj * stepInner[i];
			}
			if(i >= dc)
				break;
		}
		out[op] += val;

		// increment the kouter position
		size_t i;
		for(i = 0; i < dc; i++)
		{
			kouter[i]++;
			op += stepOuter[i];
			ip += stride * stepInner[i];
			if(kouter[i] < out.dims[i])
				break;
			op -= kouter[i] * stepOuter[i];
			ip -= kouter[i] * stride * stepInner[i];
			kouter[i] = 0;
		}
		if(i >= dc)
			break;
	}
}

// static
void Tensor::test()
{
	{
		// 1D test
		Vec in({2,3,1,0,1});
		Tensor tin(in, {5});

		Vec k({1, 0, 2});
		Tensor tk(k, {3});

		Vec out(7);
		Tensor tout(out, {7});

		Tensor::convolve(tin, tk, tout, true, 1);

		//     2 3 1 0 1
		// 2 0 1 --->
		Vec expected({2, 3, 5, 6, 3, 0, 2});
		if(std::sqrt(out.squaredDistance(expected)) > 1e-10)
			throw Ex("wrong");
	}

	{
		// 2D test
		Vec in(
			{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9
			}
		);
		Tensor tin(in, {3, 3});

		Vec k(
			{
				 1,  2,  1,
				 0,  0,  0,
				-1, -2, -1
			}
		);
		Tensor tk(k, {3, 3});

		Vec out(9);
		Tensor tout(out, {3, 3});

		Tensor::convolve(tin, tk, tout, false, 1);
		
		Vec expected(
			{
				-13, -20, -17,
				-18, -24, -18,
				 13,  20,  17
			}
		);
		if(std::sqrt(out.squaredDistance(expected)) > 1e-10)
			throw Ex("wrong");
	}
}
