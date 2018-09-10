// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef VEC_H
#define VEC_H

#include "error.h"
#include "rand.h"
#include <stdio.h>
#include <iostream>
#include <vector>


class Json;
class JsonNode;


/// Contains some useful functions for operating on vectors
class Vec
{
friend class VecWrapper;
protected:
	double* m_data;
	size_t m_size;

public:
	/// General-purpose constructor. n specifies the initial size of the vector.
	Vec(size_t n = 0);

	Vec(int n);

	/// Initializer constructor. Example usage:
	///   GVec v({2.1, 3.2, 4.0, 5.7});
	Vec(const std::initializer_list<double>& list);

	/// Copy constructor. Copies all the values in orig.
	Vec(const Vec& orig);

	/// Unmarshaling constructor
	Vec(const JsonNode& node);

	/// Copies all the values in orig.
	virtual ~Vec();

	/// Marshals this object into a JSON DOM
	virtual JsonNode* marshal(Json& doc);

	/// Returns the size of this vector.
	size_t size() const { return m_size; }

	/// Copies all the values in orig.
	void copy(const Vec& orig);

	/// Resizes this vector
	void resize(size_t n);

	/// Sets all the elements in this vector to val.
	void fill(const double val, size_t startPos = 0, size_t endPos = (size_t)-1);

	/// \brief Returns a reference to the specified element.
	inline double& operator [](size_t index) { return m_data[index]; }

	/// \brief Returns a const reference to the specified element
	inline const double& operator [](size_t index) const { return m_data[index]; }

	/// Returns a pointer to the raw element values.
	double* data() { return m_data; }

	/// Returns a const pointer to the raw element values.
	const double* data() const { return m_data; }

	/// Adds two vectors to make a new one.
	Vec operator+(const Vec& that) const;

	/// Adds another vector to this one.
	Vec& operator+=(const Vec& that);

	/// Subtracts a vector from this one to make a new one.
	Vec operator-(const Vec& that) const;

	/// Subtracts another vector from this one.
	Vec& operator-=(const Vec& that);

	/// Makes a scaled version of this vector.
	Vec operator*(double scalar) const;

	/// Scales this vector.
	Vec& operator*=(double scalar);

	/// Sets the data in this vector.
	void set(const double* pSource, size_t size);

	/// Returns the squared Euclidean magnitude of this vector.
	double squaredMagnitude() const;

	/// Scales this vector to have a magnitude of 1.0.
	void normalize();

	/// Returns the squared Euclidean distance between this and that vector.
	double squaredDistance(const Vec& that) const;

	/// Fills with random values from a uniform distribution.
	void fillUniform(Rand& rand, double min = 0.0, double max = 1.0);

	/// Fills with random values from a Normal distribution.
	void fillNormal(Rand& rand, double deviation = 1.0);

	/// Fills with random values on the surface of a sphere.
	void fillSphericalShell(Rand& rand, double radius = 1.0);

	/// Fills with random values uniformly distributed within a sphere of radius 1.
	void fillSphericalVolume(Rand& rand);

	/// Fills with random values uniformly distributed within a probability simplex.
	/// In other words, the values will sum to 1, will all be non-negative,
	/// and will not be biased toward or away from any of the extreme corners.
	void fillSimplex(Rand& rand);

	/// Prints a representation of this vector to the specified stream.
	void print(std::ostream& stream = std::cout) const;

	/// Returns the sum of the elements in this vector
	double sum() const;

	/// Returns the index of the max element.
	/// The returned value will be >= startPos.
	/// The returned value will be < endPos.
	size_t indexOfMax(size_t startPos = 0, size_t endPos = (size_t)-1) const;

	/// Returns the dot product of this and that.
	double dotProduct(const Vec& that) const;

	/// Returns the dot product of this and that, ignoring elements in which either vector has UNKNOWN_REAL_VALUE.
	double dotProductIgnoringUnknowns(const Vec& that) const;

	/// Estimates the squared distance between two points that may have some missing values. It assumes
	/// the distance in missing dimensions is approximately the same as the average distance in other
	/// dimensions. If there are no known dimensions that overlap between the two points, it returns
	/// 1e50.
	double estimateSquaredDistanceWithUnknowns(const Vec& that) const;

	/// Adds scalar * that to this vector.
	void addScaled(double scalar, const Vec& that);

	/// Applies L1 regularization to this vector.
	void regularize_L1(double amount);

	/// Puts a copy of that at the specified location in this.
	/// Throws an exception if it does not fit there.
	void put(size_t pos, const Vec& that, size_t start = 0, size_t length = (size_t)-1);

	/// Erases the specified elements. The remaining elements are shifted over.
	/// The size of the vector is decreased, but the buffer is not reallocated
	/// (so this operation wastes some memory to save a little time).
	void erase(size_t start, size_t count = 1);

	/// Returns the cosine of the angle between this and that (with the origin as the common vertex).
	double correlation(const Vec& that) const;

private:
	/// This method is deliberately private, so calling it will trigger a compiler error. Call "copy" instead.
	Vec& operator=(const Vec& orig);

	/// This method is deliberately private, so calling it will trigger a compiler error.
	Vec(double d);
};


/// Convert a Vec to a string
extern std::string to_str(const Vec& v);



/// This class temporarily wraps a Vec around an array of doubles.
/// You should take care to ensure this object is destroyed before the array it wraps.
class VecWrapper : public Vec
{
public:
	VecWrapper(Vec& vec, size_t start = 0, size_t len = (size_t)-1)
	: Vec(0)
	{
		setData(vec, start, len);
	}

	VecWrapper(double* buf = nullptr, size_t size = 0)
	: Vec(0)
	{
		setData(buf, size);
	}

	virtual ~VecWrapper()
	{
		m_data = NULL;
		m_size = 0;
	}

	void setData(Vec& vec, size_t start = 0, size_t len = (size_t)-1)
	{
		m_data = vec.data() + start;
		m_size = std::min(len, vec.size() - start);
	}

	void setData(double* buf, size_t size)
	{
		m_data = buf;
		m_size = size;
	}

	void setSize(size_t size)
	{
		m_size = size;
	}
};



/// A tensor class.
class Tensor : public VecWrapper
{
public:
	size_t* dims;
	size_t dimCount;

	/// General-purpose constructor. Example:
	/// Tensor t(v, {5, 7, 3});
	Tensor(Vec& vals, const std::initializer_list<size_t>& list);

	/// Copy constructor. Copies the dimensions. Wraps the same vector.
	Tensor(const Tensor& copyMe);

	/// Constructor for dynamically generated sizes
	Tensor(double* buf, size_t dimCount, const size_t* dims);

	/// Destructor
	virtual ~Tensor();

	/// The result is added to the existing contents of out. It does not replace the existing contents of out.
	/// Padding is computed as necessary to fill the the out tensor.
	/// filter is the filter to convolve with in.
	/// If flipFilter is true, then the filter is flipped in all dimensions.
	static void convolve(const Tensor& in, const Tensor& filter, Tensor& out, bool flipFilter = false, size_t stride = 1);

	/// Throws an exception if something is wrong.
	static void test();
};



#endif // VEC_H

