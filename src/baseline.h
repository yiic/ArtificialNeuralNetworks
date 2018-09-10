// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef BASELINE_H
#define BASELINE_H

#include <vector>
#include "matrix.h"
#include "vec.h"
#include "supervised.h"


class BaselineLearner : public SupervisedLearner
{
protected:
	Vec mode;

public:
	BaselineLearner();
	virtual ~BaselineLearner();

	/// Returns the name of this learner
	virtual const char* name() { return "Baseline"; }

	/// Train this supervised learner
	virtual void train(const Matrix& features, const Matrix& labels);

	/// Make a prediction
	virtual const Vec& predict(const Vec& in);
};


#endif // BASELINE_H
