// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "svg.h"
#include <stdlib.h>
#include "error.h"
#include "rand.h"
#include "string.h"
#include <string>
#include <sstream>
#include <cmath>

using std::string;
using std::ostringstream;


GPlotLabelSpacer::GPlotLabelSpacer(double min, double max, int maxLabels)
{
	if(maxLabels == 0)
	{
		m_spacing = 0.0;
		m_start = 0;
		m_count = 0;
		return;
	}
	if(max <= min)
		throw Ex("invalid range");
	int p = (int)ceil(log((max - min) / maxLabels) * M_LOG10E);

	// Every 10
	m_spacing = pow(10.0, p);
	m_start = (int)ceil(min / m_spacing);
	m_count = (int)floor(max / m_spacing) - m_start + 1;

	if(m_count * 5 + 4 < maxLabels)
	{
		// Every 2
		m_spacing *= 0.2;
		m_start = (int)ceil(min / m_spacing);
		m_count = (int)floor(max / m_spacing) - m_start + 1;
	}
	else if(m_count * 2 + 1 < maxLabels)
	{
		// Every 5
		m_spacing *= 0.5;
		m_start = (int)ceil(min / m_spacing);
		m_count = (int)floor(max / m_spacing) - m_start + 1;
	}
}

int GPlotLabelSpacer::count()
{
	return m_count;
}

double GPlotLabelSpacer::label(int index)
{
	return (m_start + index) * m_spacing;
}








GPlotLabelSpacerLogarithmic::GPlotLabelSpacerLogarithmic(double log_e_min, double log_e_max)
{
	double min = exp(log_e_min);
	m_max = exp(std::min(500.0, log_e_max));
	m_n = (int)floor(log_e_min * M_LOG10E);
	m_i = 1;
	while(true)
	{
		double p = pow((double)10, m_n);
		if((m_i * p) >= min)
			break;
		m_i++;
		if(m_i >= 10)
		{
			m_i = 0;
			m_n++;
		}
	}
}

bool GPlotLabelSpacerLogarithmic::next(double* pos, bool* primary)
{
	double p = pow((double)10, m_n);
	*pos = p * m_i;
	if(*pos > m_max)
		return false;
	if(m_i == 1)
		*primary = true;
	else
		*primary = false;
	m_i++;
	if(m_i >= 10)
	{
		m_i = 0;
		m_n++;
	}
	return true;
}






std::string svg_to_str(double d)
{
	std::ostringstream os;
	os.setf(std::ios::fixed);
	//os.precision(6);
	os << d;
	return os.str();
}


#define BOGUS_XMIN -1e308
const char* g_hexChars = "0123456789abcdef";

GSVG::GSVG(size_t width, size_t height, double xmin, double ymin, double xmax, double ymax, double margin)
: m_width(width), m_height(height), m_xmin(BOGUS_XMIN), m_clipping(false)
{
	m_ss << "<?xml version=\"1.0\"?><svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"";
	m_ss << to_str(width) << "\" height=\"" << to_str(height) << "\">\n";
	closeTags();
	double chartWidth = (double)m_width;
	double chartHeight = (double)m_height;
	margin = std::min(margin, 0.75 * chartWidth);
	margin = std::min(margin, 0.75 * chartHeight);
	m_hunit = ((xmax - xmin)) / (chartWidth - margin);
	m_vunit = ((ymax - ymin)) / (chartHeight - margin);
	m_margin = margin;
	m_xmin = xmin;
	m_ymin = ymin;
	m_xmax = xmax;
	m_ymax = ymax;
	m_ss << "<defs><clipPath id=\"chart0-0\"><rect x=\"" << svg_to_str(xmin) << "\" y=\"" << svg_to_str(ymin) << "\" width=\"" << svg_to_str(xmax - xmin) << "\" height=\"" << svg_to_str(ymax - ymin) << "\" /></clipPath></defs>\n";
	m_ss << "<g transform=\"translate(" << svg_to_str(margin) << " "
		<< svg_to_str(chartHeight - margin) << ") scale(" << svg_to_str((chartWidth - margin) / (xmax - xmin)) <<
		" " << svg_to_str(-(chartHeight - margin) / (ymax - ymin)) << ") translate(" << svg_to_str(-xmin) << " " << svg_to_str(-ymin) << ")\""
		<< ">\n";
}

void GSVG::closeTags()
{
	// Close the current clipping group
	if(m_clipping)
	{
		m_ss << "</g>";
		m_clipping = false;
	}

	// Close the current chart
	if(m_xmin != BOGUS_XMIN)
		m_ss << "</g>";
	m_ss << "\n\n";
}

GSVG::~GSVG()
{
}

void GSVG::clip()
{
	m_ss << "\n<!-- Clipped region -->\n";
	m_ss << "<g clip-path=\"url(#chart0-0)\">\n";
	m_clipping = true;
}

void GSVG::color(unsigned int c)
{
	m_ss << '#' << g_hexChars[(c >> 20) & 0xf] << g_hexChars[(c >> 16) & 0xf];
	m_ss << g_hexChars[(c >> 12) & 0xf] << g_hexChars[(c >> 8) & 0xf];
	m_ss << g_hexChars[(c >> 4) & 0xf] << g_hexChars[c & 0xf];
}

void GSVG::dot(double x, double y, double r, unsigned int col)
{
	m_ss << "<ellipse cx=\"" << svg_to_str(x) << "\" cy=\"" << svg_to_str(y) << "\" rx=\"" << svg_to_str(r * 4 * m_hunit) << "\" ry=\"" << svg_to_str(r * 4 * m_vunit) << "\" fill=\"";
	color(col);
	m_ss << "\" />\n";
}

void GSVG::line(double x1, double y1, double x2, double y2, double thickness, unsigned int col)
{
	m_ss << "<line x1=\"" << svg_to_str(x1) << "\" y1=\"" << svg_to_str(y1) << "\" x2=\"" << svg_to_str(x2) << "\" y2=\"" << svg_to_str(y2) << "\" style=\"stroke:";
	color(col);
	double l = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	double w = thickness * (std::abs(x2 - x1) * m_vunit + std::abs(y2 - y1) * m_hunit) / l;
	m_ss << ";stroke-width:" << svg_to_str(w) << "\"/>\n";
}

void GSVG::rect(double x, double y, double w, double h, unsigned int col)
{
	m_ss << "<rect x=\"" << svg_to_str(x) << "\" y=\"" << svg_to_str(y) << "\" width=\"" << svg_to_str(w) << "\" height=\"" << svg_to_str(h) << "\" style=\"fill:";
	color(col);
	m_ss << "\"/>\n";
}

void GSVG::text(double x, double y, const char* szText, double size, Anchor eAnchor, unsigned int col, double angle, bool serifs)
{
	double xx = x / (m_hunit * size);
	double yy = -y / (m_vunit * size);
	m_ss << "<text x=\"" << svg_to_str(xx) << "\" y=\"" << svg_to_str(yy) << "\" style=\"fill:";
	color(col);
	if(!serifs)
		m_ss << ";font-family:Sans";
	m_ss << "\" transform=\"";
	m_ss << "scale(" << svg_to_str(size * m_hunit) << " " << svg_to_str(-size * m_vunit) << ")";
	if(angle != 0.0)
		m_ss << " rotate(" << svg_to_str(-angle) << " " << svg_to_str(xx) << " " << svg_to_str(yy) << ")";
	m_ss << "\"";
	if(eAnchor == Middle)
		m_ss << " text-anchor=\"middle\"";
	else if(eAnchor == End)
		m_ss << " text-anchor=\"end\"";
	m_ss << ">" << szText << "</text>\n";
}

void GSVG::print(std::ostream& stream)
{
	closeTags();

	// Close the whole SVG file
	m_ss << "</svg>\n";

	// Print it
	stream << m_ss.str();
}

double GSVG::horizLabelPos()
{
	return m_ymin - m_vunit * ((m_margin / 2));
}

double GSVG::vertLabelPos()
{
	return m_xmin - m_hunit * ((m_margin / 2));
}

void GSVG::horizMarks(int maxLabels, std::vector<std::string>* pLabels)
{
	m_ss << "\n<!-- Horiz labels -->\n";
	if(maxLabels >= 0)
	{
		GPlotLabelSpacer spacer(m_xmin, m_xmax, maxLabels);
		int count = spacer.count();
		for(int i = 0; i < count; i++)
		{
			double x = spacer.label(i);
			line(x, m_ymin, x, m_ymax, 0.2, 0xa0a0a0);
			if(pLabels)
			{
				if(pLabels->size() > (size_t)i)
					text(x + 3 * m_hunit, m_ymin - m_vunit, (*pLabels)[i].c_str(), 1, End, 0x000000, 90);
			}
			else
				text(x + 3 * m_hunit, m_ymin - m_vunit, to_str(x).c_str(), 1, End, 0x000000, 90);
		}
	}
	else
	{
		GPlotLabelSpacerLogarithmic spacer(m_xmin, m_xmax);
		double x;
		bool primary;
		while(true)
		{
			if(!spacer.next(&x, &primary))
				break;
			line(log(x), m_ymin, log(x), m_ymax, 0.2, 0xa0a0a0);
			if(primary)
				text(log(x) + 3 * m_hunit, m_ymin - m_vunit, to_str(x).c_str(), 1, End, 0x000000, 90);
		}
	}
	m_ss << "\n";
}

void GSVG::vertMarks(int maxLabels, std::vector<std::string>* pLabels)
{
	m_ss << "\n<!-- Vert labels -->\n";
	if(maxLabels >= 0)
	{
		GPlotLabelSpacer spacer(m_ymin, m_ymax, maxLabels);
		int count = spacer.count();
		for(int i = 0; i < count; i++)
		{
			double y = spacer.label(i);
			line(m_xmin, y, m_xmax, y, 0.2, 0xa0a0a0);
			if(pLabels)
			{
				if(pLabels->size() > (size_t)i)
					text(m_xmin - m_hunit, y - 3 * m_vunit, (*pLabels)[i].c_str(), 1, End, 0x000000);
			}
			else
				text(m_xmin - m_hunit, y - 3 * m_vunit, to_str(y).c_str(), 1, End, 0x000000);
		}
	}
	else
	{
		GPlotLabelSpacerLogarithmic spacer(m_xmin, m_xmax);
		double y;
		bool primary;
		while(true)
		{
			if(!spacer.next(&y, &primary))
				break;
			line(m_xmin, log(y), m_xmax, log(y), 0.2, 0xa0a0a0);
			if(primary)
				text(m_xmin - m_hunit, log(y) - 3 * m_vunit, to_str(y).c_str(), 1, End, 0x000000);
		}
	}
	m_ss << "\n";
}


