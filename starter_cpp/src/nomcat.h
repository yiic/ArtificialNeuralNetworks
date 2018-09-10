// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef NOMCAT_H
#define NOMCAT_H

#include "preprocess.h"
#include "vec.h"



class NomCat : public PreprocessOperation
{
private:
	std::vector<size_t> m_vals;
	Matrix m_template;

public:
	NomCat() {}
	virtual ~NomCat() {}

	/// Decide how many dims are needed for each column
	virtual void train(const Matrix& data);

	/// Returns a zero-row matrix with the number of continuous columns that transform will output
	virtual const Matrix& outputTemplate() { return m_template; }

	/// Re-represent each nominal attribute with a categorical distribution of continuous values
	virtual void transform(const Vec& in, Vec& out);

	/// Re-encode categorical distributions as nominal values by finding the mode
	virtual void untransform(const Vec& in, Vec& out);
};




#endif // FILTER_H
