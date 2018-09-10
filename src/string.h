// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef string_h
#define string_h

#include <string>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <utility> //for pair
#include <ostream>
#include <sstream>


// This file provides "to_str(" functions that will convert just about anything to a std::string.


/// Convert any type that has a stream-insertion operator << to a string
template<typename T>
std::string to_str(const T& n)
{
	std::ostringstream os;
	os.precision(14);
	os << n;
	return os.str();
}

/// Convert any collection with a standard iterator to a string
template<typename T>
std::string to_str(T begin, T end)
{
	std::ostringstream os;
	os.precision(14);
	while(begin != end){ 
	  os << to_str(*begin); ++begin; 
	  if(begin != end){ os << ","; }
	}
	return os.str();
}

/// Convert a vector to a string
template<typename T>
std::string to_str(const std::vector<T>& v){
	return to_str(v.begin(), v.end());
}

/// Convert a list to a string
template<typename T>
std::string to_str(const std::list<T>& v){
	return to_str(v.begin(), v.end());
}

/// Convert a set to a string
template<typename T>
std::string to_str(const std::set<T>& v){
	return to_str(v.begin(), v.end());
}

/// Convert a deque to a string
template<typename T>
std::string to_str(const std::deque<T>& v){
	return to_str(v.begin(), v.end());
}

/// Convert a multiset to a string
template<typename T>
std::string to_str(const std::multiset<T>& v){
	return to_str(v.begin(), v.end());
}

/// Convert a multimap to a string
template<typename Key, typename T>
std::string to_str(const std::multimap<Key, T>& v){
	return to_str(v.begin(), v.end());
}

/// Convert a map to a string
template<typename Key, typename T>
std::string to_str(const std::map<Key, T>& v){
	return to_str(v.begin(), v.end());
}


#endif // string_h
