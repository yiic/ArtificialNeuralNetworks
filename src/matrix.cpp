// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "matrix.h"
#include "rand.h"
#include "error.h"
#include "string.h"
#include "vec.h"
#include "json.h"
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

using std::string;
using std::ifstream;
using std::map;
using std::vector;



Matrix::Matrix(const Matrix& other)
{
	throw Ex("Big objects should generally be passed by reference, not by value"); // (See the comment in the header file for more details.)
}

Matrix::Matrix(size_t r, size_t c)
{
	setSize(r, c);
}

Matrix::Matrix(const JsonNode& node)
{
	JsonListIterator it(&node);
	size_t r = it.remaining();
	if(r == 0)
		return;
	size_t c = 0;
	{
	JsonListIterator it2(it.current());
		c = it2.remaining();
	}
	setSize(0, c);
	while(it.remaining() > 0)
	{
		m_data.push_back(new Vec(*it.current()));
		it.advance();
	}
}

Matrix::~Matrix()
{
	clear();
}

JsonNode* Matrix::marshal(Json& doc)
{
	for(size_t i = 0; i < cols(); i++)
	{
		if(valueCount(i) > 0)
			throw Ex("Sorry, marshaling categorical values is not yet implemented");
	}
	JsonNode* pList = doc.newList();
	for(size_t i = 0; i < m_data.size(); i++)
		pList->add(&doc, row(i).marshal(doc));
	return pList;
}

void Matrix::clear()
{
	for(size_t i = 0; i < m_data.size(); i++)
		delete(m_data[i]);
	m_data.clear();
}

void Matrix::setSize(size_t rowCount, size_t columnCount)
{
	clear();

	// Set the meta-data
	m_filename = "";
	m_attr_name.resize(columnCount);
	m_str_to_enum.resize(columnCount);
	m_enum_to_str.resize(columnCount);
	for(size_t i = 0; i < columnCount; i++)
	{
		m_str_to_enum[i].clear();
		m_enum_to_str[i].clear();
	}

	newRows(rowCount);
}

void Matrix::copyMetaData(const Matrix& that)
{
	clear();
	m_attr_name = that.m_attr_name;
	m_str_to_enum = that.m_str_to_enum;
	m_enum_to_str = that.m_enum_to_str;
}

void Matrix::newColumn(size_t vals)
{
	clear();
	size_t c = cols();
	string name = "col_";
	name += to_str(c);
	m_attr_name.push_back(name);
	map <string, size_t> temp_str_to_enum;
	map <size_t, string> temp_enum_to_str;
	for(size_t i = 0; i < vals; i++)
	{
		string sVal = "val_";
		sVal += to_str(i);
		temp_str_to_enum[sVal] = i;
		temp_enum_to_str[i] = sVal;
	}
	m_str_to_enum.push_back(temp_str_to_enum);
	m_enum_to_str.push_back(temp_enum_to_str);
}

Vec& Matrix::newRow()
{
	size_t c = cols();
	if(c == 0)
		throw Ex("You must add some columns before you add any rows.");
	Vec* pNewVec = new Vec(c);
	m_data.push_back(pNewVec);
	return *pNewVec;
}

void Matrix::newRows(size_t n)
{
	m_data.reserve(m_data.size() + n);
	for(size_t i = 0; i < n; i++)
		newRow();
}

void Matrix::copy(const Matrix& that)
{
	setSize(that.rows(), that.cols());
	copyBlock(0, 0, that, 0, 0, that.rows(), that.cols());
}

double Matrix::columnMean(size_t col) const
{
	double sum = 0.0;
	size_t count = 0;
	for(size_t i = 0; i < m_data.size(); i++)
	{
		double val = (*this)[i][col];
		if(val != UNKNOWN_VALUE)
		{
			sum += val;
			count++;
		}
	}
	return sum / count;
}

double Matrix::columnMin(size_t col) const
{
	double m = 1e300;
	for(size_t i = 0; i < m_data.size(); i++)
	{
		double val = (*this)[i][col];
		if(val != UNKNOWN_VALUE)
			m = std::min(m, val);
	}
	return m;
}

double Matrix::columnMax(size_t col) const
{
	double m = -1e300;
	for(size_t i = 0; i < m_data.size(); i++)
	{
		double val = (*this)[i][col];
		if(val != UNKNOWN_VALUE)
			m = std::max(m, val);
	}
	return m;
}

double Matrix::mostCommonValue(size_t col) const
{
	map<double, size_t> counts;
	for(size_t i = 0; i < m_data.size(); i++)
	{
		double val = (*this)[i][col];
		if(val != UNKNOWN_VALUE)
		{
			map<double, size_t>::iterator pair = counts.find(val);
			if(pair == counts.end())
				counts[val] = 1;
			else
				pair->second++;
		}
	}
	size_t value_Count = 0;
	double value = 0;
	for(map<double, size_t>::iterator i = counts.begin(); i != counts.end(); i++)
	{
		if(i->second > value_Count)
		{
			value = i->first;
			value_Count = i->second;
		}
	}
	return value;
}

void Matrix::copyBlock(size_t destRow, size_t destCol, const Matrix& that, size_t rowBegin, size_t colBegin, size_t rowCount, size_t colCount)
{
	if (destRow + rowCount > rows() || destCol + colCount > cols())
		throw Ex("Out of range for destination matrix.");
	if(rowBegin + rowCount > that.rows() || colBegin + colCount > that.cols())
		throw Ex("Out of range for source matrix.");

	// Copy the specified region of meta-data
	for(size_t i = 0; i < colCount; i++)
	{
		m_attr_name[destCol + i] = that.m_attr_name[colBegin + i];
		m_str_to_enum[destCol + i] = that.m_str_to_enum[colBegin + i];
		m_enum_to_str[destCol + i] = that.m_enum_to_str[colBegin + i];
	}

	// Copy the specified region of data
	for(size_t i = 0; i < rowCount; i++)
		(*this)[destRow + i].put(destCol, that[rowBegin + i], colBegin, colCount);
}

string toLower(string strToConvert)
{
	//change each element of the string to lower case
	for(size_t i = 0; i < strToConvert.length(); i++)
		strToConvert[i] = tolower(strToConvert[i]);
	return strToConvert;//return the converted string
}

void Matrix::saveARFF(string filename) const
{
	std::ofstream s;
	s.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		s.open(filename.c_str(), std::ios::binary);
	}
	catch(const std::exception&)
	{
		throw Ex("Error creating file: ", filename);
	}
	s.precision(10);
	s << "@RELATION " << m_filename << "\n";
	for(size_t i = 0; i < m_attr_name.size(); i++)
	{
		s << "@ATTRIBUTE " << m_attr_name[i];
		if(m_attr_name[i].size() == 0)
			s << "x";
		size_t vals = valueCount(i);
		if(vals == 0)
			s << " REAL\n";
		else
		{
			s << " {";
			for(size_t j = 0; j < vals; j++)
			{
				s << attrValue(i, j);
				if(j + 1 < vals)
					s << ",";
			}
			s << "}\n";
		}
	}
	s << "@DATA\n";
	for(size_t i = 0; i < rows(); i++)
	{
		const Vec& r = (*this)[i];
		for(size_t j = 0; j < cols(); j++)
		{
			if(r[j] == UNKNOWN_VALUE)
				s << "?";
			else
			{
				size_t vals = valueCount(j);
				if(vals == 0)
					s << to_str(r[j]);
				else
				{
					size_t val = (size_t)r[j];
					if(val >= vals)
						throw Ex("value out of range");
					s << attrValue(j, val);
				}
			}
			if(j + 1 < cols())
				s << ",";
		}
		s << "\n";
	}
}

void Matrix::loadARFF(string fileName)
{
	size_t lineNum = 0;
	string line;                 //line of input from the arff file
	ifstream inputFile;          //input stream
	map <string, size_t> tempMap;   //temp map for int->string map (attrInts)
	map <size_t, string> tempMapS;  //temp map for string->int map (attrString)
	size_t attrCount = 0;           //Count number of attributes
	clear();
	inputFile.open ( fileName.c_str() );
	if ( !inputFile )
		throw Ex ( "failed to open the file: ", fileName );
	while ( !inputFile.eof() && inputFile )
	{
		//Iterate through each line of the file
		getline ( inputFile, line );
		lineNum++;
		if ( toLower ( line ).find ( "@relation" ) == 0 )
			m_filename = line.substr ( line.find_first_of ( " " ) );
		else if ( toLower ( line ).find ( "@attribute" ) == 0 )
		{
			line = line.substr ( line.find_first_of ( " \t" ) + 1 );
			
			string attr_name;
			
			// If the attribute name is delimited by ''
			if ( line.find_first_of( "'" ) == 0 )
			{
				attr_name = line.substr ( 1 );
				attr_name = attr_name.substr ( 0, attr_name.find_first_of( "'" ) );
				line = line.substr ( attr_name.size() + 2 );
			}
			else
			{
				attr_name = line.substr( 0, line.find_first_of( " \t" ) );
				line = line.substr ( attr_name.size() );
			}
			
			m_attr_name.push_back ( attr_name );
			
			string value = line.substr ( line.find_first_not_of ( " \t" ) );
			
			tempMap.clear();
			tempMapS.clear();

			//If the attribute is nominal
			if ( value.find_first_of ( "{" ) == 0 )
			{
				size_t firstComma;
				size_t firstSpace;
				size_t firstLetter;
				value = value.substr ( 1, value.find_last_of ( "}" ) - 1 );
				size_t valCount = 0;
				string tempValue;

				//Parse the attributes--push onto the maps
				while ( ( firstComma = value.find_first_of ( "," ) ) != string::npos )
				{
					firstLetter = value.find_first_not_of ( " \t," );

					value = value.substr ( firstLetter );
					firstComma = value.find_first_of ( "," );
					firstSpace = value.find_first_of ( " \t" );
					tempMapS[valCount] = value.substr ( 0, firstComma );
					string valName = value.substr ( 0, firstComma );
					valName = valName.substr ( 0, valName.find_last_not_of(" \t") + 1);
					tempMap[valName] = valCount++;
					firstComma = ( firstComma < firstSpace &&
						firstSpace < ( firstComma + 2 ) ) ? firstSpace :
						firstComma;
					value = value.substr ( firstComma + 1 );
				}

				//Push final attribute onto the maps
				firstLetter = value.find_first_not_of ( " \t," );
				value = value.substr ( firstLetter );
				string valName = value.substr ( 0, value.find_last_not_of(" \t") + 1);
				tempMapS[valCount] = valName;
				tempMap[valName] = valCount++;
				m_str_to_enum.push_back ( tempMap );
				m_enum_to_str.push_back ( tempMapS );
			}
			else
			{
				//The attribute is continuous
				m_str_to_enum.push_back ( tempMap );
				m_enum_to_str.push_back ( tempMapS );
			}
			attrCount++;
		}
		else if ( toLower ( line ).find ( "@data" ) == 0 )
		{
			while ( !inputFile.eof() )
			{
				getline ( inputFile, line );
				lineNum++;
				if(line.length() == 0 || line[0] == '%' || line[0] == '\n' || line[0] == '\r')
					continue;
				size_t pos = 0;
				Vec& newrow = newRow();
				for ( size_t i = 0; i < attrCount; i++ )
				{
					size_t vals = valueCount ( i );
					size_t valStart = line.find_first_not_of ( " \t", pos );
					if(valStart == string::npos)
						throw Ex("Expected more elements on line ", to_str(lineNum));
					size_t valEnd = line.find_first_of ( ",\n\r", valStart );
					string val;
					if(valEnd == string::npos)
					{
						if(i + 1 == attrCount)
							val = line.substr( valStart );
						else
							throw Ex("Expected more elements on line ", to_str(lineNum));
					}
					else
						val = line.substr ( valStart, valEnd - valStart );
					pos = valEnd + 1;
					if ( vals > 0 ) //if the attribute is nominal...
					{
						if ( val == "?" )
							newrow[i] = UNKNOWN_VALUE;
						else
						{
							map<string, size_t>::iterator it = m_str_to_enum[i].find ( val );
							if(it == m_str_to_enum[i].end())
								throw Ex("Unrecognized enumeration value, \"", val, "\" on line ", to_str(lineNum), ", attr ", to_str(i)); 
							newrow[i] = (double)m_str_to_enum[i][val];
						}
					}
					else
					{
						// The attribute is continuous
						if ( val == "?" )
							newrow[i] = UNKNOWN_VALUE;
						else
							newrow[i] = atof( val.c_str() );
					}
				}
			}
		}
	}
}

const std::string& Matrix::attrValue(size_t attr, size_t val) const
{
	std::map<size_t, std::string>::const_iterator it = m_enum_to_str[attr].find(val);
	if(it == m_enum_to_str[attr].end())
		throw Ex("no name");
	return it->second;
}

void Matrix::fill(double val)
{
	for(size_t i = 0; i < m_data.size(); i++)
		m_data[i]->fill(val);
}

void Matrix::checkCompatibility(const Matrix& that) const
{
	size_t c = cols();
	if(that.cols() != c)
		throw Ex("Matrices have different number of columns");
	for(size_t i = 0; i < c; i++)
	{
		if(valueCount(i) != that.valueCount(i))
			throw Ex("Column ", to_str(i), " has mis-matching number of values");
	}
}

void Matrix::print(std::ostream& stream)
{
	for(size_t i = 0; i < rows(); i++)
	{
		stream << "[" << row(i)[0];
		for(size_t j = 1; j < cols(); j++)
			stream << "," << row(i)[j];
		stream << "]\n";
	}
}

Matrix* Matrix::transpose()
{
	size_t r = rows();
	size_t c = cols();
	Matrix* pTarget = new Matrix(c, r);
	for(size_t i = 0; i < c; i++)
	{
		for(size_t j = 0; j < r; j++)
			pTarget->row(i)[j] = row(j)[i];
	}
	return pTarget;
}

void Matrix::swapRows(size_t a, size_t b)
{
	std::swap(m_data[a], m_data[b]);
}

void Matrix::swapColumns(size_t nAttr1, size_t nAttr2)
{
	if(nAttr1 == nAttr2)
		return;
	std::swap(m_attr_name[nAttr1], m_attr_name[nAttr2]);
	std::swap(m_str_to_enum[nAttr1], m_str_to_enum[nAttr2]);
	std::swap(m_enum_to_str[nAttr1], m_enum_to_str[nAttr2]);
	size_t nCount = rows();
	for(size_t i = 0; i < nCount; i++)
	{
		Vec& r = row(i);
		std::swap(r[nAttr1], r[nAttr2]);
	}
}

// static
Matrix* Matrix::multiply(const Matrix& a, const Matrix& b, bool transposeA, bool transposeB)
{
	if(transposeA)
	{
		if(transposeB)
		{
			size_t dims = a.rows();
			if((size_t)b.cols() != dims)
				throw Ex("dimension mismatch");
			size_t w = b.rows();
			size_t h = a.cols();
			Matrix* pOut = new Matrix(h, w);
			for(size_t y = 0; y < h; y++)
			{
				Vec& r = pOut->row(y);
				for(size_t x = 0; x < w; x++)
				{
					const Vec& pB = b[x];
					double sum = 0;
					for(size_t i = 0; i < dims; i++)
						sum += a[i][y] * pB[i];
					r[x] = sum;
				}
			}
			return pOut;
		}
		else
		{
			size_t dims = a.rows();
			if(b.rows() != dims)
				throw Ex("dimension mismatch");
			size_t w = b.cols();
			size_t h = a.cols();
			Matrix* pOut = new Matrix(h, w);
			for(size_t y = 0; y < h; y++)
			{
				Vec& r = pOut->row(y);
				for(size_t x = 0; x < w; x++)
				{
					double sum = 0;
					for(size_t i = 0; i < dims; i++)
						sum += a[i][y] * b[i][x];
					r[x] = sum;
				}
			}
			return pOut;
		}
	}
	else
	{
		if(transposeB)
		{
			size_t dims = (size_t)a.cols();
			if((size_t)b.cols() != dims)
				throw Ex("dimension mismatch");
			size_t w = b.rows();
			size_t h = a.rows();
			Matrix* pOut = new Matrix(h, w);
			for(size_t y = 0; y < h; y++)
			{
				Vec& r = pOut->row(y);
				const Vec& pA = a[y];
				for(size_t x = 0; x < w; x++)
				{
					const Vec& pB = b[x];
					double sum = 0;
					for(size_t i = 0; i < dims; i++)
						sum += pA[i] * pB[i];
					r[x] = sum;
				}
			}
			return pOut;
		}
		else
		{
			size_t dims = (size_t)a.cols();
			if(b.rows() != dims)
				throw Ex("dimension mismatch");
			size_t w = b.cols();
			size_t h = a.rows();
			Matrix* pOut = new Matrix(h, w);
			for(size_t y = 0; y < h; y++)
			{
				Vec& r = pOut->row(y);
				const Vec& pA = a[y];
				for(size_t x = 0; x < w; x++)
				{
					double sum = 0;
					for(size_t i = 0; i < dims; i++)
						sum += pA[i] * b[i][x];
					r[x] = sum;
				}
			}
			return pOut;
		}
	}
}

Matrix& Matrix::operator*=(double scalar)
{
	for(size_t i = 0; i < rows(); i++)
	{
		Vec& r = row(i);
		for(size_t j = 0; j < cols(); j++)
			r[j] *= scalar;
	}
	return *this;
}


void verboten()
{
	throw Ex("Tried to copy a holder. (The method that facilitates this should have been private to catch this at compile time.)");
}

template <class T>
class Holder
{
private:
	T* m_p;

public:
	Holder(T* p = NULL)
	{
		//COMPILER_ASSERT(sizeof(T) > sizeof(double));
		m_p = p;
	}

private:
	/// Private copy constructor so that the attempts to copy a
	/// holder are caught at compile time rather than at run time
	Holder(const Holder& other)
	{
		verboten();
	}
public:
	/// Deletes the object that is being held
	~Holder()
	{
		delete(m_p);
	}
private:
	/// Private operator= so that the attempts to copy a
	/// holder are caught at compile time rather than at run time
	const Holder& operator=(const Holder& other)
	{
		verboten();
		return *this;
	}
public:
	/// Deletes the object that is being held, and sets the holder
	/// to hold p.  Will not delete the held pointer if the new
	/// pointer is the same.
	void reset(T* p = NULL)
	{
		if(p != m_p)
		{
			delete(m_p);
			m_p = p;
		}
	}

	/// Returns a pointer to the object being held
	T* get()
	{
		return m_p;
	}

	/// Releases the object. (After calling this method, it is your job to delete the object.)
	T* release()
	{
		T* pTmp = m_p;
		m_p = NULL;
		return pTmp;
	}

	T& operator*() const
	{
		return *m_p;
	}

	T* operator->() const
	{
		return m_p;
	}
};

template <class T>
class ArrayHolder
{
private:
	T* m_p;

public:
	ArrayHolder(T* p = NULL)
	{
		m_p = p;
	}
private:	
	ArrayHolder(const ArrayHolder& other)
	{
		verboten();
	}
public:	
	/// Deletes the array of objects being held
	~ArrayHolder()
	{
		delete[] m_p;
	}
private:
	const ArrayHolder& operator=(const ArrayHolder& other)
	{
		verboten();
		return *this;
	}
public:
	/// Deletes the array of objects being held and sets this holder to hold NULL
	void reset(T* p = NULL)
	{
		if(p != m_p)
		{
			delete[] m_p;
			m_p = p;
		}
	}

	/// Returns a pointer to the first element of the array being held
	T* get()
	{
		return m_p;
	}

	/// Releases the array. (After calling this method, it is your job to delete the array.)
	T* release()
	{
		T* pTmp = m_p;
		m_p = NULL;
		return pTmp;
	}

	T& operator[](size_t n)
	{
		return m_p[n];
	}
};


double Matrix_pythag(double a, double b)
{
	double at = std::abs(a);
	double bt = std::abs(b);
	if(at > bt)
	{
		double ct = bt / at;
		return at * sqrt(1.0 + ct * ct);
	}
	else if(bt > 0.0)
	{
		double ct = at / bt;
		return bt * sqrt(1.0 + ct * ct);
	}
	else
		return 0.0;
}

double Matrix_takeSign(double a, double b)
{
	return (b >= 0.0 ? std::abs(a) : -std::abs(a));
}

double Matrix_safeDivide(double n, double d)
{
	if(d == 0.0 && n == 0.0)
		return 0.0;
	else
	{
		double t = n / d;
		//GAssert(t > -1e200, "prob");
		return t;
	}
}

void Matrix::fixNans()
{
	size_t colCount = cols();
	for(size_t i = 0; i < rows(); i++)
	{
		Vec& r = row(i);
		for(size_t j = 0; j < colCount; j++)
		{
			if(r[j] >= -1e308 && r[j] < 1e308)
			{
			}
			else
				r[j] = (i == (size_t)j ? 1.0 : 0.0);
		}
	}
}

void Matrix::singularValueDecompositionHelper(Matrix** ppU, double** ppDiag, Matrix** ppV, bool throwIfNoConverge, size_t maxIters)
{
	int m = (int)rows();
	int n = (int)cols();
	if(m < n)
		throw Ex("Expected at least as many rows as columns");
	int j, k;
	int l = 0;
	int p, q;
	double c, f, h, s, x, y, z;
	double norm = 0.0;
	double g = 0.0;
	double scale = 0.0;
	Matrix* pU = new Matrix(m, m);
	Holder<Matrix> hU(pU);
	pU->fill(0.0);
	for(int i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
			pU->row(i)[j] = row(i)[j];
	}
	double* pSigma = new double[n];
	ArrayHolder<double> hSigma(pSigma);
	Matrix* pV = new Matrix(n, n);
	Holder<Matrix> hV(pV);
	pV->fill(0.0);
	double* temp = new double[n];
	ArrayHolder<double> hTemp(temp);

	// Householder reduction to bidiagonal form
	for(int i = 0; i < n; i++)
	{
		// Left-hand reduction
		temp[i] = scale * g;
		l = i + 1;
		g = 0.0;
		s = 0.0;
		scale = 0.0;
		if(i < m)
		{
			for(k = i; k < m; k++)
				scale += std::abs(pU->row(k)[i]);
			if(scale != 0.0)
			{
				for(k = i; k < m; k++)
				{
					pU->row(k)[i] = Matrix_safeDivide(pU->row(k)[i], scale);
					double t = pU->row(k)[i];
					s += t * t;
				}
				f = pU->row(i)[i];
				g = -Matrix_takeSign(sqrt(s), f);
				h = f * g - s;
				pU->row(i)[i] = f - g;
				if(i != n - 1)
				{
					for(j = l; j < n; j++)
					{
						s = 0.0;
						for(k = i; k < m; k++)
							s += pU->row(k)[i] * pU->row(k)[j];
						f = Matrix_safeDivide(s, h);
						for(k = i; k < m; k++)
							pU->row(k)[j] += f * pU->row(k)[i];
					}
				}
				for(k = i; k < m; k++)
					pU->row(k)[i] *= scale;
			}
		}
		pSigma[i] = scale * g;

		// Right-hand reduction
		g = 0.0;
		s = 0.0;
		scale = 0.0;
		if(i < m && i != n - 1)
		{
			for(k = l; k < n; k++)
				scale += std::abs(pU->row(i)[k]);
			if(scale != 0.0)
			{
				for(k = l; k < n; k++)
				{
					pU->row(i)[k] = Matrix_safeDivide(pU->row(i)[k], scale);
					double t = pU->row(i)[k];
					s += t * t;
				}
				f = pU->row(i)[l];
				g = -Matrix_takeSign(sqrt(s), f);
				h = f * g - s;
				pU->row(i)[l] = f - g;
				for(k = l; k < n; k++)
					temp[k] = Matrix_safeDivide(pU->row(i)[k], h);
				if(i != m - 1)
				{
					for(j = l; j < m; j++)
					{
						s = 0.0;
						for(k = l; k < n; k++)
							s += pU->row(j)[k] * pU->row(i)[k];
						for(k = l; k < n; k++)
							pU->row(j)[k] += s * temp[k];
					}
				}
				for(k = l; k < n; k++)
					pU->row(i)[k] *= scale;
			}
		}
		norm = std::max(norm, std::abs(pSigma[i]) + std::abs(temp[i]));
	}

	// Accumulate right-hand transform
	for(int i = n - 1; i >= 0; i--)
	{
		if(i < n - 1)
		{
			if(g != 0.0)
			{
				for(j = l; j < n; j++)
					pV->row(i)[j] = Matrix_safeDivide(Matrix_safeDivide(pU->row(i)[j], pU->row(i)[l]), g); // (double-division to avoid underflow)
				for(j = l; j < n; j++)
				{
					s = 0.0;
					for(k = l; k < n; k++)
						s += pU->row(i)[k] * pV->row(j)[k];
					for(k = l; k < n; k++)
						pV->row(j)[k] += s * pV->row(i)[k];
				}
			}
			for(j = l; j < n; j++)
			{
				pV->row(i)[j] = 0.0;
				pV->row(j)[i] = 0.0;
			}
		}
		pV->row(i)[i] = 1.0;
		g = temp[i];
		l = i;
	}

	// Accumulate left-hand transform
	for(int i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = pSigma[i];
		if(i < n - 1)
		{
			for(j = l; j < n; j++)
				pU->row(i)[j] = 0.0;
		}
		if(g != 0.0)
		{
			g = Matrix_safeDivide(1.0, g);
			if(i != n - 1)
			{
				for(j = l; j < n; j++)
				{
					s = 0.0;
					for(k = l; k < m; k++)
						s += pU->row(k)[i] * pU->row(k)[j];
					f = Matrix_safeDivide(s, pU->row(i)[i]) * g;
					for(k = i; k < m; k++)
						pU->row(k)[j] += f * pU->row(k)[i];
				}
			}
			for(j = i; j < m; j++)
				pU->row(j)[i] *= g;
		}
		else
		{
			for(j = i; j < m; j++)
				pU->row(j)[i] = 0.0;
		}
		pU->row(i)[i] += 1.0;
	}

	// Diagonalize the bidiagonal matrix
	for(k = n - 1; k >= 0; k--) // For each singular value
	{
		for(size_t iter = 1; iter <= maxIters; iter++)
		{
			// Test for splitting
			bool flag = true;
			for(l = k; l >= 0; l--)
			{
				q = l - 1;
				if(std::abs(temp[l]) + norm == norm)
				{
					flag = false;
					break;
				}
				if(std::abs(pSigma[q]) + norm == norm)
					break;
			}

			if(flag)
			{
				c = 0.0;
				s = 1.0;
				for(int i = l; i <= k; i++)
				{
					f = s * temp[i];
					temp[i] *= c;
					if(std::abs(f) + norm == norm)
						break;
					g = pSigma[i];
					h = Matrix_pythag(f, g);
					pSigma[i] = h;
					h = Matrix_safeDivide(1.0, h);
					c = g * h;
					s = -f * h;
					for(j = 0; j < m; j++)
					{
						y = pU->row(j)[q];
						z = pU->row(j)[i];
						pU->row(j)[q] = y * c + z * s;
						pU->row(j)[i] = z * c - y * s;
					}
				}
			}

			z = pSigma[k];
			if(l == k)
			{
				// Detect convergence
				if(z < 0.0)
				{
					// Singular value should be positive
					pSigma[k] = -z;
					for(j = 0; j < n; j++)
						pV->row(k)[j] *= -1.0;
				}
				break;
			}
			if(throwIfNoConverge && iter >= maxIters)
				throw Ex("failed to converge");

			// Shift from bottom 2x2 minor
			x = pSigma[l];
			q = k - 1;
			y = pSigma[q];
			g = temp[q];
			h = temp[k];
			f = Matrix_safeDivide(((y - z) * (y + z) + (g - h) * (g + h)), (2.0 * h * y));
			g = Matrix_pythag(f, 1.0);
			f = Matrix_safeDivide(((x - z) * (x + z) + h * (Matrix_safeDivide(y, (f + Matrix_takeSign(g, f))) - h)), x);

			// QR transform
			c = 1.0;
			s = 1.0;
			for(j = l; j <= q; j++)
			{
				int i = j + 1;
				g = temp[i];
				y = pSigma[i];
				h = s * g;
				g = c * g;
				z = Matrix_pythag(f, h);
				temp[j] = z;
				c = Matrix_safeDivide(f, z);
				s = Matrix_safeDivide(h, z);
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for(p = 0; p < n; p++)
				{
					x = pV->row(j)[p];
					z = pV->row(i)[p];
					pV->row(j)[p] = x * c + z * s;
					pV->row(i)[p] = z * c - x * s;
				}
				z = Matrix_pythag(f, h);
				pSigma[j] = z;
				if(z != 0.0)
				{
					z = Matrix_safeDivide(1.0, z);
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;
				for(p = 0; p < m; p++)
				{
					y = pU->row(p)[j];
					z = pU->row(p)[i];
					pU->row(p)[j] = y * c + z * s;
					pU->row(p)[i] = z * c - y * s;
				}
			}
			temp[l] = 0.0;
			temp[k] = f;
			pSigma[k] = x;
		}
	}

	// Sort the singular values from largest to smallest
	for(int i = 1; i < n; i++)
	{
		for(j = i; j > 0; j--)
		{
			if(pSigma[j - 1] >= pSigma[j])
				break;
			pU->swapColumns(j - 1, j);
			pV->swapRows(j - 1, j);
			std::swap(pSigma[j - 1], pSigma[j]);
		}
	}

	// Return results
	pU->fixNans();
	pV->fixNans();
	*ppU = hU.release();
	*ppDiag = hSigma.release();
	*ppV = hV.release();
}

Matrix* Matrix::pseudoInverse()
{
	Matrix* pU;
	double* pDiag;
	Matrix* pV;
	size_t colCount = cols();
	size_t rowCount = rows();
	if(rowCount < (size_t)colCount)
	{
		Matrix* pTranspose = transpose();
		Holder<Matrix> hTranspose(pTranspose);
		pTranspose->singularValueDecompositionHelper(&pU, &pDiag, &pV, false, 80);
	}
	else
		singularValueDecompositionHelper(&pU, &pDiag, &pV, false, 80);
	Holder<Matrix> hU(pU);
	ArrayHolder<double> hDiag(pDiag);
	Holder<Matrix> hV(pV);
	Matrix sigma(rowCount < (size_t)colCount ? colCount : rowCount, rowCount < (size_t)colCount ? rowCount : colCount);
	sigma.fill(0.0);
	size_t m = std::min(rowCount, colCount);
	for(size_t i = 0; i < m; i++)
	{
		if(std::abs(pDiag[i]) > 1e-9)
			sigma[i][i] = Matrix_safeDivide(1.0, pDiag[i]);
		else
			sigma[i][i] = 0.0;
	}
	Matrix* pT = Matrix::multiply(*pU, sigma, false, false);
	Holder<Matrix> hT(pT);
	if(rowCount < (size_t)colCount)
		return Matrix::multiply(*pT, *pV, false, false);
	else
		return Matrix::multiply(*pV, *pT, true, true);
}





std::string to_str(const Matrix& v)
{
	std::ostringstream os;
	for(size_t i = 0; i < v.rows(); i++)
	{
		if(i == 0)
			os << "[";
		else
			os << " ";
		os << to_str(v[i]);
		os << "\n";
	}
	os << "]";
	return os.str();
}
