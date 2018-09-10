// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "nomcat.h"
#include "error.h"
#include "string.h"
#include <math.h>



// virtual
void NomCat::train(const Matrix& data)
{
	size_t totalVals = 0;
	m_vals.resize(data.cols());
	for(size_t i = 0; i < data.cols(); i++)
	{
		size_t n = data.valueCount(i);
		if(n < 3)
			n = 1;
		m_vals[i] = n;
		totalVals += n;
	}
	m_template.setSize(0, totalVals);
}

// virtual
void NomCat::transform(const Vec& in, Vec& out)
{
	if(in.size() != m_vals.size())
		throw Ex("NomCat::transform received unexpected in-vector size. Expected ", to_str(m_vals.size()), ", got ", to_str(in.size()));
	out.resize(m_template.cols());
	size_t outPos = 0;
	for(size_t i = 0; i < in.size(); i++)
	{
		if(m_vals[i] == 1)
			out[outPos++] = in[i];
		else
		{
			size_t outStart = outPos;
			for(size_t j = 0; j < m_vals[i]; j++)
				out[outPos++] = 0.0;
			if(in[i] != UNKNOWN_VALUE)
			{
				if(in[i] >= m_vals[i])
					throw Ex("Value out of range. Expected [0-", to_str(m_vals[i] - 1), "], got ", to_str(in[i]));
				out[outStart + (size_t)in[i]] = 1.0;
			}
		}
	}
}

// virtual
void NomCat::untransform(const Vec& in, Vec& out)
{
	if(in.size() != m_template.cols())
		throw Ex("NomCat::untransform received unexpected in-vector size. Expected ", to_str(m_template.cols()), ", got ", to_str(in.size()));
	out.resize(m_vals.size());
	size_t inPos = 0;
	for(size_t i = 0; i < m_vals.size(); i++)
	{
		if(m_vals[i] == 1)
			out[i] = in[inPos++];
		else
		{
			size_t startIn = inPos;
			size_t maxIndex = 0;
			inPos++;
			for(size_t j = 1; j < m_vals[i]; j++)
			{
				if(in[inPos] > in[startIn + maxIndex])
					maxIndex = inPos - startIn;
				inPos++;
			}
			out[i] = (double)maxIndex;
		}
	}
}


