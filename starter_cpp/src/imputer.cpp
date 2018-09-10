// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "imputer.h"
#include "error.h"
#include "string.h"
#include <math.h>



// virtual
void Imputer::train(const Matrix& data)
{
	m_template.copyMetaData(data);
	m_centroid.resize(data.cols());
	for(size_t i = 0; i < data.cols(); i++)
	{
		if(data.valueCount(i) == 0)
			m_centroid[i] = data.columnMean(i);
		else
			m_centroid[i] = data.mostCommonValue(i);
	}
}

// virtual
void Imputer::transform(const Vec& in, Vec& out)
{
	if(in.size() != m_centroid.size())
		throw Ex("Imputer::transform received unexpected in-vector size. Expected ", to_str(m_centroid.size()), ", got ", to_str(in.size()));
	out.resize(m_centroid.size());
	for(size_t i = 0; i < m_centroid.size(); i++)
	{
		if(in[i] == UNKNOWN_VALUE)
			out[i] = m_centroid[i];
		else
			out[i] = in[i];
	}
}

// virtual
void Imputer::untransform(const Vec& in, Vec& out)
{
	if(in.size() != m_centroid.size())
		throw Ex("Imputer::untransform received unexpected in-vector size. Expected ", to_str(m_centroid.size()), ", got ", to_str(in.size()));
	out.resize(m_centroid.size());
	for(size_t i = 0; i < m_centroid.size(); i++)
		out[i] = in[i];
}


