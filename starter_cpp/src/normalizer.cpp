// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "normalizer.h"
#include "string.h"
#include "error.h"
#include <algorithm>



// virtual
void Normalizer::train(const Matrix& data)
{
	m_template.copyMetaData(data);
	m_inputMins.resize(data.cols());
	m_inputMaxs.resize(data.cols());
	for(size_t i = 0; i < data.cols(); i++)
	{
		if(data.valueCount(i) == 0)
		{
			// Compute the min and max
			m_inputMins[i] = data.columnMin(i);
			m_inputMaxs[i] = std::max(m_inputMins[i] + 1e-9, data.columnMax(i));
		}
		else
		{
			// Don't do nominal attributes
			m_inputMins[i] = UNKNOWN_VALUE;
			m_inputMaxs[i] = UNKNOWN_VALUE;
		}
	}
}

// virtual
void Normalizer::transform(const Vec& in, Vec& out)
{
	if(in.size() != m_inputMins.size())
		throw Ex("Normalizer::transform received unexpected in-vector size. Expected ", to_str(m_inputMins.size()), ", got ", to_str(in.size()));
	out.resize(in.size());
	for(size_t c = 0; c < in.size(); c++)
	{
		if(m_inputMins[c] == UNKNOWN_VALUE) // if the attribute is nominal...
			out[c] = in[c];
		else
		{
			if(in[c] == UNKNOWN_VALUE)
				out[c] = UNKNOWN_VALUE;
			else
				out[c] = (in[c] - m_inputMins[c]) / (m_inputMaxs[c] - m_inputMins[c]);
		}
	}
}

// virtual
void Normalizer::untransform(const Vec& in, Vec& out)
{
	if(in.size() != m_inputMins.size())
		throw Ex("Normalizer::untransform received unexpected in-vector size. Expected ", to_str(m_inputMins.size()), ", got ", to_str(in.size()));
	out.resize(in.size());
	for(size_t c = 0; c < in.size(); c++)
	{
		if(m_inputMins[c] == UNKNOWN_VALUE) // if the attribute is nominal...
			out[c] = in[c];
		else
		{
			if(in[c] == UNKNOWN_VALUE)
				out[c] = UNKNOWN_VALUE;
			else
				out[c] = in[c] * (m_inputMaxs[c] - m_inputMins[c]) + m_inputMins[c];
		}
	}
}


