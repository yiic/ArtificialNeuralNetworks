// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "matrix.h"
#include <vector>


class Vec;

class PreprocessOperation
{
public:
	PreprocessOperation() {}
	virtual ~PreprocessOperation() {}

	/// Trains this unsupervised learner
	virtual void train(const Matrix& data) = 0;

	/// Returns an example of an output matrix. The meta-data of this matrix
	/// shows how output will be given. (This matrix contains no data because it has zero rows.
	/// It is only used for the meta-data.)
	virtual const Matrix& outputTemplate() = 0;

	/// Transform a single instance. (Assumes train was previously called.)
	virtual void transform(const Vec& in, Vec& out) = 0;

	/// Untransform a single instance. (Assumes train was previously called.)
	virtual void untransform(const Vec& in, Vec& out) = 0;

	/// Applies the transform to each row in the specified matrix. (Assumes train was previously called.)
	Matrix* transformBatch(const Matrix& in);
};



#endif // PREPROCESS_H
