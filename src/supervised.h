// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef SUPERVISED_H
#define SUPERVISED_H

#include <vector>
#include "matrix.h"



class SupervisedLearner
{
public:
	SupervisedLearner() {}
	virtual ~SupervisedLearner() {};

	/// Returns the name of this learner
	virtual const char* name() = 0;

	/// Train this learner
	virtual void train(const Matrix& features, const Matrix& labels) = 0;

	/// Partially train using a single pattern
	virtual void trainIncremental(const Vec& feat, const Vec& lab);

	/// Make a prediction
	virtual const Vec& predict(const Vec& in) = 0;

	/// Measures the misclassifications with the provided test data
	size_t countMisclassifications(const Matrix& features, const Matrix& labels);

	/// This default implementation just copies the data, without changing it in any way.
	virtual void filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out);
	double sumSquaredError(const Matrix& features, const Matrix& labels);
	double NFoldCrossVal(Matrix& features, Matrix& labels, int folds, int iter);
};


#endif // SUPERVISED_H
