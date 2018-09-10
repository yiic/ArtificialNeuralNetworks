/*
  The contents of this file are dedicated to the public domain
  (http://creativecommons.org/publicdomain/zero/1.0/).
*/

#ifndef MYIMAGE_H
#define MYIMAGE_H

#include <stddef.h>

/// Returns the blue channel value from a 32-bit ARGB pixel value.
#define gBlue(c) ((c) & 0xff)

/// Returns the green channel value from a 32-bit ARGB pixel value.
#define gGreen(c) (((c) >> 8) & 0xff)

/// Returns the red channel value from a 32-bit ARGB pixel value.
#define gRed(c) (((c) >> 16) & 0xff)

/// Returns the alpha channel value from a 32-bit ARGB pixel value.
#define gAlpha(c) (((c) >> 24) & 0xff)

/// Combines alpha, red, green, and blue channel values (each from 0-255) into a single 32-bit ARGB pixel value.
/// For an opaque image, the alpha value should be 0xff.
inline unsigned int gARGB(int a, int r, int g, int b)
{
	return ((b & 0xff) | ((g & 0xff) << 8) | ((r & 0xff) << 16) | ((a & 0xff) << 24));
}

/// A simple image class
class MyImage
{
public:
	unsigned int* m_pPixels;
	unsigned int m_width;
	unsigned int m_height;

	/// Constructor
	MyImage();

	/// Destructor
	virtual ~MyImage();

	/// Change the size of this image.
	/// The raw pixels are initialized with garbage.
	void resize(unsigned int w, unsigned int h);

	/// Load the image from a PNG file.
	void loadPng(const char* szFilename);

	/// Save the image as a PNG to a file.
	void savePng(const char* szFilename);

	/// Get the pixel, p, at location (x,y) in 32-bit ARGB format.
	/// Behavior is undefined if x or y is out of range.
	inline unsigned int pixel(int x, int y) const
	{
		return m_pPixels[m_width * y + x];
	}

	/// Set a pixel
	/// Behavior is undefined if x or y is out of range.
	inline void setPixel(int x, int y, unsigned int color)
	{
		m_pPixels[m_width * y + x] = color;
	}
};

#endif // MYIMAGE_H
