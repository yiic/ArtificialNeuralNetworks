// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef IMPUTER_H
#define IMPUTER_H

#include "preprocess.h"
#include "vec.h"


/// Replaces missing continuous values with the mean
/// Replaces missing categorical values with the most-common value (or mode)
class Imputer : public PreprocessOperation
{
private:
	Vec m_centroid;
	Matrix m_template;

public:
	Imputer() {}
	virtual ~Imputer() {}

	/// Computes the column means of continuous attributes, and the mode of nominal attributes
	virtual void train(const Matrix& data);

	/// Returns a zero-row matrix with the same column meta-data as the one that was passed to train.
	virtual const Matrix& outputTemplate() { return m_template; }

	/// Replace missing values with the centroid value
	virtual void transform(const Vec& in, Vec& out);

	/// Copy in to out. (In other words, this is a no-op.)
	virtual void untransform(const Vec& in, Vec& out);
};




#endif // IMPUTER_H
