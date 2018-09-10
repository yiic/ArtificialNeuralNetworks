// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef __PLOT_H__
#define __PLOT_H__

#include <vector>
#include <string>
#include <sstream>


class GImage;
class GRand;


/// If you need to place grid lines or labels at regular intervals
/// (like 1000, 2000, 3000, 4000... or 20, 25, 30, 35... or 0, 2, 4, 6, 8, 10...)
/// this class will help you pick where to place the labels so that
/// there are a reasonable number of them, and they all land on nice label
/// values.
class GPlotLabelSpacer
{
protected:
	double m_spacing;
	int m_start;
	int m_count;

public:
	/// maxLabels specifies the maximum number of labels that it can ever
	/// decide to use. (It should be just smaller than the number of labels
	/// that would make the graph look too crowded.)
	GPlotLabelSpacer(double min, double max, int maxLabels);

	/// Returns the number of labels that have been picked. It will be a value
	/// smaller than maxLabels.
	int count();

	/// Returns the location of the n'th label (where 0 <= n < count())
	double label(int index);
};


/// Similar to GPlotLabelSpacer, except for logarithmic grids. To plot in
/// logarithmic space, set your plot window to have a range from log_e(min)
/// to log_e(max). When you actually plot things, plot them at log_e(x), where
/// x is the position of the thing you want to plot.
class GPlotLabelSpacerLogarithmic
{
protected:
	double m_max;
	int m_n, m_i;

public:
	/// Pass in the log (base e) of your min and max values. (We make you
	/// pass them in logarithmic form, so you can't use a negative min value.)
	GPlotLabelSpacerLogarithmic(double log_e_min, double log_e_max);

	/// Returns true and sets *pos to the position of the next label.
	/// (You should actually plot it at log_e(*pos) in your plot window.)
	/// Returns false if there are no more (and doesn't set *pos).
	/// primary is set to true if the label is the primary
	/// label for the new scale.
	bool next(double* pos, bool* primary);
};


/// This class simplifies plotting data to an SVG file
class GSVG
{
public:
	enum Anchor
	{
		Start,
		Middle,
		End,
	};

protected:
	std::stringstream m_ss;
	size_t m_width, m_height, m_hPos, m_vPos;
	double m_hunit, m_vunit, m_margin;
	double m_xmin, m_ymin, m_xmax, m_ymax;
	bool m_clipping;

public:
	/// This object represents a hWindows-by-vWindows grid of charts.
	/// width and height specify the width and height of the entire grid of charts.
	/// xmin, ymin, xmax, and ymax specify the coordinates in the chart to begin drawing.
	/// margin specifies the size of the margin for axis labels.
	GSVG(size_t width = 1024, size_t height = 768, double xmin = 0, double ymin = 0, double xmax = 80, double ymax = 50, double margin = 50);
	~GSVG();

	/// Returns (xmax - xmin) / width, which is often a useful size.
	double hunit() { return m_hunit; }

	/// Returns (ymax - ymin) / height, which is often a useful size.
	double vunit() { return m_vunit; }

	/// Draw a dot
	void dot(double x, double y, double r = 1.0, unsigned int col = 0x000080);

	/// Draw a line
	void line(double x1, double y1, double x2, double y2, double thickness = 1.0, unsigned int col = 0x008000);

	/// Draw a rectangle
	void rect(double x, double y, double w, double h, unsigned int col = 0x008080);

	/// Draw text
	void text(double x, double y, const char* szText, double size = 1.0, Anchor eAnchor = Start, unsigned int col = 0x000000, double angle = 0.0, bool serifs = true);

	/// Generate an SVG file with all of the components that have been added so far.
	void print(std::ostream& stream);

	/// Label the horizontal axis. If maxLabels is 0, then no grid-lines will be drawn. If maxLabels is -1, then
	/// Logarithmic grid-lines will be drawn. If pLabels is non-NULL, then its values will be used to label
	/// the grid-lines instead of the continuous values.
	void horizMarks(int maxLabels, std::vector<std::string>* pLabels = NULL);

	/// Label the vertical axis. If maxLabels is 0, then no grid-lines will be drawn. If maxLabels is -1, then
	/// Logarithmic grid-lines will be drawn. If pLabels is non-NULL, then its values will be used to label
	/// the grid-lines instead of the continuous values.
	void vertMarks(int maxLabels, std::vector<std::string>* pLabels = NULL);

	/// Returns a good y position for the horizontal axis label
	double horizLabelPos();

	/// Returns a good x position for the vertical axis label
	double vertLabelPos();

	/// After calling this method, all draw operations will be clipped to fall within (xmin, ymin)-(xmax, ymax),
	/// until a new chart is started.
	void clip();

protected:
	void color(unsigned int c);
	void closeTags();
};



#endif // __PLOT_H__
