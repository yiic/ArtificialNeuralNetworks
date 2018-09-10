// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "preprocess.h"

Matrix* PreprocessOperation::transformBatch(const Matrix& in)
{
	Matrix* out = new Matrix();
	out->copyMetaData(outputTemplate());
	out->newRows(in.rows());
	for(size_t i = 0; i < in.rows(); i++)
		transform(in[i], (*out)[i]);
	return out;
}
