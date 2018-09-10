// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef NORMALIZER_H
#define NORMALIZER_H

#include "preprocess.h"
#include "vec.h"


class Normalizer : public PreprocessOperation
{
private:
	Vec m_inputMins;
	Vec m_inputMaxs;
	Matrix m_template;

public:
	Normalizer() {}
	virtual ~Normalizer() {}

	// Computes the min and max of each column
	virtual void train(const Matrix& data);

	// Returns a zero-row matrix with the same column meta-data as the one that was passed to train.
	virtual const Matrix& outputTemplate() { return m_template; }

	// Normalize continuous features to fall in the 
	virtual void transform(const Vec& in, Vec& out);

	// De-normalize continuous values
	virtual void untransform(const Vec& in, Vec& out);
};



#endif // NORMALIZER_H
